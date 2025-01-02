from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import numpy as np
import wave
import threading
import time
import os
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import sys
import logging
from logging.handlers import RotatingFileHandler
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import json
import werkzeug
import uuid
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from audio_extract import extract_audio


sentry_sdk.init(
    dsn="https://cb8037c1a5f827203638c77a613105b0@o4508518491226112.ingest.de.sentry.io/4508518580486224",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    _experiments={
        # Set continuous_profiling_auto_start to True
        # to automatically start the profiler on when
        # possible.
        "continuous_profiling_auto_start": True,
    }, environment="production",
    integrations=[FlaskIntegration()]

)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1024 * 1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# Configure Gemini

gemini_api = "AIzaSyDuRZPOikRxilF7k5zoMRoHGzNyfP6EKMc"
genai.configure(api_key=gemini_api, transport='rest')
model = genai.GenerativeModel('gemini-2.0-flash-exp')

app = Flask(__name__)
CORS(app)

# Configure directories
VOICE_DIR = "voice"
TRANSCRIPTION_DIR = "transcriptions"
SUMMARY_DIR = "summaries"
for dir in [VOICE_DIR, TRANSCRIPTION_DIR, SUMMARY_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Configure request queue and thread pool
REQUEST_QUEUE_SIZE = 10
request_queue = Queue(maxsize=REQUEST_QUEUE_SIZE)
thread_pool = ThreadPoolExecutor(max_workers=3)


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
        try:
            timestamp = int(time.time() * 1000)
            timestamp_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y%m%d_%H%M%S')

            # Save file with timestamp
            unique_id = str(uuid.uuid4())
            new_filepath = os.path.join(VOICE_DIR, f"{unique_id}_{original_filename}")
            os.rename(file_path, new_filepath)

            # Get transcription
            transcription = self._transcribe_audio_with_retry(new_filepath)

            # Save transcription
            transcription_filepath = os.path.join(TRANSCRIPTION_DIR, f"{unique_id}_transcription.json")
            self._save_transcription(transcription, transcription_filepath)

            #summary_result = self.summarize_text(transcription, timestamp_str)
            #if not summary_result['success']:
            #    return jsonify(summary_result), 500

            return {
                'success': True,
                'Summary Notes': transcription,
                #'summary': summary_result,
                'filepath': new_filepath,
                'transcription_filepath': transcription_filepath,
                'timestamp': timestamp
            }

        except Exception as e:
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

    def summarize_text(self, text: str, timestamp_str: str) -> dict:
        if self.lang == 'ar':
            language = 'arabic'
        else:
            language = 'english'

        try:
            if not model:
                return {
                    'success': False,
                    'error': "Summarization service unavailable"
                }

            summarization_prompt = f""" This is a business meeting 
                you role is an experienced minute taker

                THE INPUT: you will be given a meeting and you should do your job as the most experienced minute taker do

                The Expected output is : all important information, dates, decisions , tasks and deadlines mentioned in the meeting. 
                ensure documentation of the decisions and actions taken in the meeting, which facilitates their follow-up and implementation

                Please provide a comprehensive summary of the audio content. NOT ALL content just summarize the meeting in the points i have told you above
                Focus on the main points discussed and key takeaways.
                Format the summary in clear paragraphs with proper punctuation. result should be in {language}.
            """

            response = model.generate_content([summarization_prompt, text])
            summary = response.text

            # Save summary to file
            summary_filepath = os.path.join(SUMMARY_DIR, f"{timestamp_str}_summary.json")
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': summary,
                    'timestamp': time.time(),
                    'original_timestamp': timestamp_str
                }, f, ensure_ascii=False)

            return {
                'success': True,
                'summary': summary,
                'summary_filepath': summary_filepath
            }
        except Exception as e:
            logger.error(f"Summarization error: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)
            return {
                'success': False,
                'error': "Summarization failed"
            }


# Create audio processor instance
audio_processor = AudioProcessor()


def process_queued_request(file_path: str, original_filename: str):
    result = audio_processor.process_audio_file(file_path, original_filename)
    return result


@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if request_queue.full():
            return jsonify({
                'success': False,
                'error': 'Server is busy. Please try again later.'
            }), 503

        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400



        # Save uploaded file temporarily
        temp_filepath = os.path.join(VOICE_DIR, werkzeug.utils.secure_filename(file.filename))
        file.save(temp_filepath)
        if not file.filename.endswith('.wav'):
            input_video=temp_filepath
            file_type = str(temp_filepath).split('.')[-1]
            output_audio = temp_filepath.replace(file_type, 'wav')
            extract_audio(input_path=input_video,
                          output_path=output_audio,
                          output_format='wav')
            temp_filepath = output_audio
        # Process file through queue
        future = thread_pool.submit(process_queued_request, temp_filepath, file.filename)
        result = future.result()

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error handling file upload: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Serve files from voice directory
app.add_url_rule('/voice/<path:filename>', endpoint='voice_files',
                 view_func=lambda filename: send_from_directory('voice', filename))

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        thread_pool.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sentry_sdk.capture_exception(e)
    finally:
        logger.info("Cleanup complete")
