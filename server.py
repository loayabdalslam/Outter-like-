from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
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

# Configure SocketIO with larger message size and ping settings
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,  # 100MB buffer
    async_handlers=True
)

# Configure directories
VOICE_DIR = "voice"
OUTPUT_DIR = "output"
for dir in [VOICE_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)

class AudioProcessor:
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_process_time = time.time()
        self.recordings = []
        self.lock = threading.Lock()
        self.transcription_errors = 0
        self.max_retries = 3

    def process_audio_chunk(self, audio_data: np.ndarray, timestamp: int, mode: str):
        with self.lock:
            try:
                self.audio_buffer = np.append(self.audio_buffer, audio_data)
                current_time = time.time()

                if current_time - self.last_process_time >= 5:
                    if len(self.audio_buffer) > 0:
                        self._process_buffer(timestamp)
                    self.last_process_time = current_time

            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                self.audio_buffer = np.array([], dtype=np.float32)

    def _process_buffer(self, timestamp: int):
        try:
            timestamp_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y%m%d_%H%M%S')
            
            if len(self.audio_buffer) > 0:
                filepath = self._save_audio(self.audio_buffer, f"{timestamp_str}_recording.wav")
                self.recordings.append({"path": filepath, "timestamp": timestamp})
                self._process_recording(filepath, timestamp)
                self.audio_buffer = np.array([], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error in _process_buffer: {e}", exc_info=True)

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

    def _process_recording(self, filepath: str, timestamp: int):
        try:
            transcription = self._transcribe_audio_with_retry(filepath)
            if transcription:
                socketio.emit('transcription', {
                    'text': transcription,
                    'timestamp': timestamp,
                    'mode': 'recording'
                })
        except Exception as e:
            logger.error(f"Error processing recording: {e}", exc_info=True)

    def _transcribe_audio_with_retry(self, audio_path: str, retries=0) -> str:
        try:
            if not model:
                return "Transcription service unavailable"
                
            audio_file = genai.upload_file(path=audio_path)
            prompt = """
            Please transcribe this audio file and provide a clear, well-formatted transcription.
            The audio is a WAV file containing speech that needs to be transcribed accurately.
            Please maintain natural speech patterns and include proper punctuation. result should be in arabic
            """
            response = model.generate_content([prompt, audio_file])
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

    def generate_summary(self) -> str:
        with self.lock:
            try:
                if not self.recordings:
                    return "No recordings available to summarize"

                merged_path = os.path.join(OUTPUT_DIR, "merged_audio.wav")
                self._merge_recordings(merged_path)
                
                if not model:
                    return "Summary service unavailable"
                    
                audio_file = genai.upload_file(path=merged_path)
                prompt = """
                Please provide a comprehensive summary of the audio content.
                Focus on the main points discussed and key takeaways.
                Format the summary in clear paragraphs with proper punctuation. result should be in arabic.
                """
                response = model.generate_content([prompt, audio_file])
                return response.text
                
            except Exception as e:
                logger.error(f"Error generating summary: {e}", exc_info=True)
                return f"Error generating summary: {str(e)}"

    def _merge_recordings(self, output_path: str):
        if not self.recordings:
            raise ValueError("No recordings to merge")

        try:
            with wave.open(self.recordings[0]["path"], 'rb') as first_wav:
                params = first_wav.getparams()
                
            merged_frames = bytearray()
            for recording in sorted(self.recordings, key=lambda x: x["timestamp"]):
                with wave.open(recording["path"], 'rb') as wav_file:
                    merged_frames.extend(wav_file.readframes(wav_file.getnframes()))

            with wave.open(output_path, 'wb') as output_wav:
                output_wav.setparams(params)
                output_wav.writeframes(merged_frames)
        except Exception as e:
            logger.error(f"Error merging recordings: {e}", exc_info=True)
            raise

# Create audio processor instance
audio_processor = AudioProcessor()

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('audio_data')
def handle_audio_data(data):
    try:
        # Convert base64 string to numpy array if needed
        if isinstance(data['buffer'], str):
            buffer_data = base64.b64decode(data['buffer'])
            audio_array = np.frombuffer(buffer_data, dtype=np.float32)
        else:
            audio_array = np.frombuffer(data['buffer'], dtype=np.float32)
            
        timestamp = data.get('timestamp', int(time.time() * 1000))
        mode = data.get('mode', 'recording')
        
        if len(audio_array) > 0:
            logger.debug(f"Received audio chunk - Length: {len(audio_array)}, Mode: {mode}")
            audio_processor.process_audio_chunk(audio_array, timestamp, mode)
    except Exception as e:
        logger.error(f"Error handling audio data: {e}", exc_info=True)

@socketio.on('generate_summary')
def handle_generate_summary():
    try:
        summary = audio_processor.generate_summary()
        socketio.emit('summary_result', {'text': summary})
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        socketio.emit('summary_result', {'error': str(e)})

@app.route('/voice/<path:filename>')
def serve_voice_files(filename):
    return send_from_directory('voice', filename)

if __name__ == '__main__':
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
    finally:
        logger.info("Cleanup complete")
