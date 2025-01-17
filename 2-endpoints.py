import json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import logging
from logging.handlers import RotatingFileHandler
import google.generativeai as genai
from dotenv import load_dotenv
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

import werkzeug
import threading
from moviepy.editor import AudioFileClip
from tinydb import TinyDB, Query

# Initialize TinyDB
db = TinyDB('database.json', encoding='utf-8', indent=4, ensure_ascii=False, sort_keys=True, separators=(',', ': '))
db_lock = threading.Lock()
File = Query()

# Initialize Sentry for error tracking
sentry_sdk.init(
    dsn="https://cb8037c1a5f827203638c77a613105b0@o4508518491226112.ingest.de.sentry.io/4508518580486224",
    traces_sample_rate=1.0,
    _experiments={"continuous_profiling_auto_start": True},
    environment="production",
    integrations=[FlaskIntegration()]
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1024 * 1024, backupCount=5,encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
gemini_api = "AIzaSyDuRZPOikRxilF7k5zoMRoHGzNyfP6EKMc"
genai.configure(api_key=gemini_api, transport='rest')
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure directories
UPLOAD_DIR = "uploads"
WAV_DIR = "wav_files"
TRANSCRIPTION_DIR = "transcriptions"
for dir in [UPLOAD_DIR, WAV_DIR, TRANSCRIPTION_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)

class AudioProcessor:
    def __init__(self, lang='ar'):
        self.language = lang
        self.transcription_prompt = " "

    def transcribe_audio(self, audio_path: str,lang) -> str:
        try:
            self.language=lang
            self.transcription_prompt = f"""This is a business meeting
          you role is an experienced minute taker

          THE INPUT: you will be given a meeting and you should do your job as the most experienced minute taker do

          The Expected output is : all important information, dates, decisions , tasks and deadlines mentioned in the meeting.
          ensure documentation of the decisions and actions taken in the meeting, which facilitates their follow-up and implementation.Format the summary in clear paragraphs with proper punctuation. result should be in language {self.language}  and in html format.

          Please provide a comprehensive summary of the audio content. NOT ALL content just summarize the meeting in the points i have told you
          above.   Focus on the main points discussed and key takeaways, and in addition to that add another Section that has
          the very important details of the meeting and their corresponding explanations in a seperated list to make sure we cover
          everything but make sure that this section is the last section and it doesn't change the structure we agreed on
          for important information mentioned above.
                                    """


            audio_file = genai.upload_file(path=audio_path)
            response = model.generate_content([self.transcription_prompt, audio_file])
            return response.text
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            sentry_sdk.capture_exception(e)
            return "Transcription failed"

# Create audio processor instance
audio_processor = AudioProcessor()

# Endpoint 1: Upload and convert file
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify(success=False, error="No file provided"), 400

        file = request.files['file']
        lang = request.form.get('lang')
        logger.info(f" !!!!!!!!!!!  transcribtion at: {lang}")
        print(type(lang))
        if file.filename == '':
            return jsonify(success=False, error="No file selected"), 400
        if lang is None:  # Check if 'lang' is None
            lang = 'of the speaker (what is in arabic make it in arabic and what is in english make it in english)'

        #language of the meeting
        language=lang.lower()
        # Generate a unique ID for the file
        unique_id = str(uuid.uuid4())
        original_filename = werkzeug.utils.secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_DIR, f"{unique_id}_{original_filename}")
        wav_path = os.path.join(WAV_DIR, f"{unique_id}.wav")
        input_file = upload_path
        output_file = wav_path
        # Save the uploaded file
        file.save(upload_path)
        logger.info(f"File saved temporarily at: {upload_path}")

        # Convert to WAV format
        if not original_filename.endswith('.wav'):
            logger.info(f"Converting file to WAV: {upload_path} -> {wav_path}")



            video = AudioFileClip(input_file)
            video.write_audiofile(output_file)


        else:


            os.rename(upload_path, wav_path)
            logger.info(f"File is already in WAV format: {output_file}")

        # Store file information in the database
        with db_lock:
            db.insert({'file_id': unique_id, 'wav_path': output_file, 'summary': None})

        # Automatically call the second endpoint in a background thread
        app_context = app.app_context()

        threading.Thread(target=run_summary_in_thread, args=(app_context, unique_id,language)).start()

        return jsonify(success=True, file_id=unique_id), 200

    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        return jsonify(success=False, error=str(e)), 500

def run_summary_in_thread(app_context, unique_id,language):
    with app_context:  # Push the application context to the thread
        summarize_file(language,unique_id)


# Endpoint 2: Summarize the WAV file
@app.route('/summarize', methods=['POST'])
def summarize_file(language,file_id: str = None):
    try:

        # Retrieve the file path from the database
        result = db.search(File.file_id == file_id)
        if not result:
            logger.error(f"No file found with ID: {file_id}")
            return

        wav_path = result[0]['wav_path']

        if not os.path.exists(wav_path):
            logger.error(f"WAV file not found: {wav_path}")
            return
        # Transcribe and summarize the WAV file
        lang=language
        logger.info(f"Starting transcription for file: {wav_path}")
        summary = audio_processor.transcribe_audio(wav_path,lang)
        summary = summary.replace('\n', ' ')
        summary = summary.replace('**', ' ')
        logger.info(f"Transcription completed for file: {wav_path}")
        # Update the summary in the database
        with db_lock:
            db.update({'summary': summary}, File.file_id == file_id)
        logger.info(f"Summary updated for file: {file_id}")

        # Save transcription
        transcription_path = os.path.join(TRANSCRIPTION_DIR, f"{file_id}_transcription.json")
        with open(transcription_path, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary}, f, ensure_ascii=False)
        logger.info(f"Transcription saved at: {transcription_path}")

        # Clean up temporary files
        if os.path.exists(wav_path):
            os.remove(wav_path)
            logger.info(f"WAV file removed: {wav_path}")
        print(f"Summary for file {file_id}: {summary}")
        return jsonify(success=True, summary=summary), 200

    except Exception as e:
        logger.error(f"Error summarizing file: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)
        return jsonify(success=False, error=str(e)), 500

# Serve files from WAV directory
@app.route('/wav/<path:filename>')
def serve_wav(filename):
    return send_from_directory(WAV_DIR, filename)

## Adding a GET endpoint that allows users to check the summarization status.
@app.route('/status/<file_id>', methods=['GET'])
def get_status(file_id):
    result = db.search(File.file_id == file_id)
    if result:
        summary = result[0].get('summary')
        if summary:
            return jsonify(success=True, summary=summary), 200
        else:
            return jsonify(success=True, status="Processing"), 200
    else:
        return jsonify(success=False, error="File not found"), 404


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sentry_sdk.capture_exception(e)
    finally:
        logger.info("Cleanup complete")