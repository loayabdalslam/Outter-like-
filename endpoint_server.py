from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import os
import uuid
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from audio_extract import extract_audio
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import google.generativeai as genai
from datetime import datetime
global unique_id

gemini_api = "AIzaSyDuRZPOikRxilF7k5zoMRoHGzNyfP6EKMc"
genai.configure(api_key=gemini_api, transport='rest')
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Directories for file handling
VOICE_DIR = "voice"
TRANSCRIPTION_DIR = "transcriptions"
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1024 * 1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Sentry for error tracking
sentry_sdk.init(
    dsn="https://example_sentry_dsn@ingest.sentry.io/1234567",
    traces_sample_rate=1.0,
    integrations=[FlaskIntegration()]
)

# Configure request queue and thread pool
REQUEST_QUEUE_SIZE = 10
request_queue = Queue(maxsize=REQUEST_QUEUE_SIZE)
thread_pool = ThreadPoolExecutor(max_workers=3)

# In-memory status store for simplicity (use a database in production)
task_status = {}

class AudioProcessor:
    def __init__(self, lang='ar'):
        self.lang = lang
        self.lock = threading.Lock()
        self.transcription_errors = 0
        self.max_retries = 3

        if lang == 'ar':
            language = 'arabic'
        else:
            language = 'english'

        self.transcription_prompt = f"""This is a business meeting
          you role is an experienced minute taker

          THE INPUT: you will be given a meeting and you should do your job as the most experienced minute taker do

          The Expected output is : all important information, dates, decisions , tasks and deadlines mentioned in the meeting.
          ensure documentation of the decisions and actions taken in the meeting, which facilitates their follow-up and implementation

          Please provide a comprehensive summary of the audio content. NOT ALL content just summarize the meeting in the points i have told you 
          above.   Focus on the main points discussed and key takeaways, and in addition to that add another Section that has
          the very important details of the meeting and their corresponding explanations in a seperated list to make sure we cover 
          everything but make sure that this section is the last section and it doesn't change the structure we agreed on
          for important information mentioned above. Format the summary in clear paragraphs with proper punctuation. result should be in {language}.
                                    """

    def process_audio_file(self, file_path: str, original_filename: str) -> dict:
        unique_id = str(uuid.uuid4())
        try:
            timestamp = int(time.time() * 1000)
            timestamp_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y%m%d_%H%M%S')
            print(f"file path to check : {file_path}")
            # Save file with timestamp
            new_filepath = str(VOICE_DIR) + '\\' + str(f"{unique_id}_{original_filename}")
            file_type = new_filepath.split('.')[-1]
            new_filepath = new_filepath.replace(file_type,'wav')
            os.rename(file_path, new_filepath)

            # Get transcription
            transcription = self._transcribe_audio_with_retry(new_filepath)

            # Save transcription
            transcription_filepath = os.path.join(TRANSCRIPTION_DIR, f"{unique_id}_transcription.json")
            self._save_transcription(transcription, transcription_filepath)


            return {
                'success': True,
                'Summary Notes': transcription,
                'filepath': new_filepath,
                'transcription_filepath': transcription_filepath,
                'timestamp': timestamp
            }

        except Exception as e:
            new_filepath = os.path.join(VOICE_DIR, f"{unique_id}_{original_filename}")
            print(f"filepath : {new_filepath}")
            logger.error(f"Error processing audio file: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)
            return {
                'success': False,
                'error': str(e)
            }

    def _save_transcription(self, transcription: str, filepath: str):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({'Summary notes': transcription, 'timestamp': time.time()}, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving transcription file: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)
            raise

    def _transcribe_audio_with_retry(self, audio_path: str, retries=0) -> str:
        try:
            if not model:
                return "Transcription service unavailable"

            print(f'audio path for upload: {audio_path}')
            audio_file = genai.upload_file(path=audio_path)
            response = model.generate_content([self.transcription_prompt, audio_file])
            self.transcription_errors = 0  # Reset error count on success
            return response.text
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)
            self.transcription_errors += 1

            if retries < self.max_retries and self.transcription_errors < 5:
                logger.info(f"Retrying transcription, attempt {retries + 1}")
                time.sleep(1)  # Wait before retrying
                return self._transcribe_audio_with_retry(audio_path, retries + 1)
            else:
                return "Transcription temporarily unavailable"

processor = AudioProcessor()

def process_queued_request(file_path: str, original_filename: str):
    result = processor.process_audio_file(file_path, original_filename)
    return result

def handle_audio_processing(task_id, file_path, filename):
    try:
        summary = processor.process_audio_file(file_path, filename)
        summary_path = os.path.join(TRANSCRIPTION_DIR, f"{task_id}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({'summary': summary}, f)
        task_status[task_id] = {'status': 'completed', 'summary': summary}
    except Exception as e:
        task_status[task_id] = {'status': 'error', 'error': str(e)}

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    task_id = unique_id
    temp_filepath = os.path.join(VOICE_DIR, f"{task_id}_{file.filename}")
    file.save(temp_filepath)

    task_status[task_id] = {'status': 'pending'}
    thread_pool.submit(handle_audio_processing, task_id, temp_filepath, file.filename)

    return jsonify({'success': True, 'task_id': task_id})

@app.route('/transcriptions/<task_id>', methods=['GET'])
def get_summary(task_id):
    if task_id not in task_status:
        return jsonify({'success': False, 'error': 'Invalid task ID'}), 404

    status = task_status[task_id]
    if status['status'] == 'pending':
        return jsonify({'success': True, 'status': 'pending'}), 202
    elif status['status'] == 'completed':
        return jsonify({'success': True, 'status': 'completed', 'summary': status['summary']}), 200
    else:
        return jsonify({'success': False, 'error': status.get('error', 'Unknown error')}), 500

@app.route('/voice/<path:filename>', methods=['GET'])
def serve_voice_file(filename):
    return send_from_directory(VOICE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
