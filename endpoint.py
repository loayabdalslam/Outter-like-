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

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# Configure Gemini
genai.configure(api_key='AIzaSyD3DIAlu69Amj0o6UKm3fhORJ3HGOdAEik', transport='rest')
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
    def __init__(self):
        self.lock = threading.Lock()
        self.transcription_errors = 0
        self.max_retries = 3
        self.transcription_prompt = """
            Please transcribe this audio file and provide a clear, well-formatted transcription.
            The audio is a WAV file containing speech that needs to be transcribed accurately.
            pUT IN YOUR CONCERNS TO DEFINE THE SPEAKER, I DON'T MEAN DIARIZATION,but i need an inform that a different speaker is here, transcribe the meeting until different one speak and then split it , and start the transcribtion of the next and so on ALSO FORMAT THE TRANSCRIBE AS A dialogue with timestamps
            Please maintain natural speech patterns and include proper punctuation. result should be in arabic
            """

    def process_audio(self, audio_data: bytes, timestamp: int) -> dict:
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            timestamp_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y%m%d_%H%M%S')
            filepath = self._save_audio(audio_array, f"{timestamp_str}_recording.wav")
            transcription = self._transcribe_audio_with_retry(filepath)
            
            # Save transcription to file
            transcription_filepath = os.path.join(TRANSCRIPTION_DIR, f"{timestamp_str}_transcription.json")
            self._save_transcription(transcription, transcription_filepath)
            
            return {
                'success': True,
                'transcription': transcription,
                'filepath': filepath,
                'transcription_filepath': transcription_filepath,
                'timestamp': timestamp
            }
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def _save_audio(self, buffer: np.ndarray, filename: str) -> str:
        filepath = os.path.join(VOICE_DIR, filename)
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                audio_16bit = (buffer * 32767).astype(np.int16)
                wav_file.writeframes(audio_16bit.tobytes())
            return filepath
        except Exception as e:
            logger.error(f"Error saving audio file: {e}", exc_info=True)
            raise

    def _save_transcription(self, transcription: str, filepath: str):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({'transcription': transcription, 'timestamp': time.time()}, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving transcription file: {e}", exc_info=True)
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
            self.transcription_errors += 1
            
            if retries < self.max_retries and self.transcription_errors < 5:
                logger.info(f"Retrying transcription, attempt {retries + 1}")
                time.sleep(1)  # Wait before retrying
                return self._transcribe_audio_with_retry(audio_path, retries + 1)
            else:
                return f"Transcription temporarily unavailable"

def summarize_text(text: str, timestamp_str: str) -> dict:
    try:
        if not model:
            return {
                'success': False,
                'error': "Summarization service unavailable"
            }
        
        
        summarization_prompt = """ This is a business meeting 
                you role is an experienced minute taker
                

                THE INPUT: you will be given a meeting and you should do your job as the most experienced minute taker do

                The Expected output is : all important information, dates, decisions , tasks and deadlines mentioned in the meeting. ensure documentation of the decisions and actions taken in the meeting, which facilitates their follow-up and implementation

                Please provide a comprehensive summary of the audio content. NOT ALL content just summarize the meeting in the points i have told you above
                Focus on the main points discussed and key takeaways.
                Format the summary in clear paragraphs with proper punctuation. result should be in arabic.
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
        return {
            'success': False,
            'error': "Summarization failed"
        }

# Create audio processor instance
audio_processor = AudioProcessor()

def process_queued_request(audio_data, timestamp):
    result = audio_processor.process_audio(audio_data, timestamp)
    return result

@app.route('/process-audio', methods=['POST'])
def handle_audio_data():
    try:
        if request_queue.full():
            return jsonify({
                'success': False,
                'error': 'Server is busy. Please try again later.'
            }), 503

        audio_data = base64.b64decode(request.json['buffer'])
        timestamp = request.json.get('timestamp', int(time.time() * 1000))
        
        # Add request to queue and process asynchronously
        future = thread_pool.submit(process_queued_request, audio_data, timestamp)
        result = future.result()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error handling audio data: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/summarize/<path:filename>', methods=['GET'])
def get_summary():
    try:
        current_directory = os.getcwd()
        
        folder_path = os.path.join(current_directory, TRANSCRIPTION_DIR)
        folder_contents = os.listdir(folder_path)
        filename = folder_contents[0]
        # Get timestamp from filename
        timestamp_str = filename.split('_')[0]
        
        # Check if summary already exists
        summary_filename = f"{timestamp_str}_summary.json"
        summary_filepath = os.path.join(SUMMARY_DIR, summary_filename)
        
        if os.path.exists(summary_filepath):
            with open(summary_filepath, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                return jsonify({
                    'success': True,
                    'summary': summary_data['summary'],
                    'summary_filepath': summary_filepath,
                    'cached': True
                })
        
        # If no summary exists, generate one from the transcription
        transcription_filename = f"{timestamp_str}_transcription.json"
        transcription_filepath = os.path.join(TRANSCRIPTION_DIR, transcription_filename)
        
        if not os.path.exists(transcription_filepath):
            return jsonify({
                'success': False,
                'error': 'Transcription not found'
            }), 404

        # Load transcription from file
        with open(transcription_filepath, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
            transcription = transcription_data['transcription']

        # Generate and save summary
        summary_result = summarize_text(transcription, timestamp_str)
        
        if not summary_result['success']:
            return jsonify(summary_result), 500
            
        return jsonify(summary_result)
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/voice/<path:filename>')
def serve_voice_files(filename):
    return send_from_directory('voice', filename)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        thread_pool.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
    finally:
        logger.info("Cleanup complete")