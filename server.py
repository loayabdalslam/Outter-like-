from flask import Flask,send_from_directory
from flask_cors import CORS
import logging
from flask_socketio import SocketIO
import numpy as np
import wave
import io
import threading
import queue
import time
import os
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import asyncio

# Load environment variables
load_dotenv()

# Configure Google Gemini
genai.configure(api_key="AIzaSyA-Ayj7gvkSjdFdwlFBgUcRWTz9lPYEb5U")
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Add socket.io logging
logging.getLogger('socketio').setLevel(logging.DEBUG)
logging.getLogger('engineio').setLevel(logging.DEBUG)

# Add this after app initialization
VOICE_DIR = "voice"
if not os.path.exists(VOICE_DIR):
    os.makedirs(VOICE_DIR)

class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        self.voice_dir = VOICE_DIR
        self.last_process_time = time.time()
        self.buffer = np.array([], dtype=np.float32)
        self.loop = asyncio.new_event_loop()
 


    async def process_with_gemini(self, audio_file_path: str) -> str:
        """Process audio file directly with Gemini AI"""

        if not os.path.isfile(audio_file_path):
            raise ValueError(f"Invalid audio file path: {audio_file_path}")

        try:
            audio_file = genai.upload_file(path=audio_file_path)
            # Create prompt for Gemini
            prompt = f"""
            Please transcribe this audio file and provide a clear, well-formatted transcription.
            The audio is a WAV file containing speech that needs to be transcribed accurately.
            Please maintain natural speech patterns and include proper punctuation.
            """

            # Send to Gemini with audio data - add await here
            response =  model.generate_content([prompt, audio_file])
            print('Transcription: ', response.text)
            # Get the text directly from response
            return response.text
        except (IOError, ConnectionError) as e:
            logger.error(f"Error processing audio with Gemini: {e} (File: {audio_file_path})")
            return "Error processing audio"
        except Exception as e:
            logger.exception(f"Unexpected error processing audio: {e}")
            return ""
    
    
    def process_audio_chunk(self, audio_data: np.ndarray, timestamp: int, mode: str):
        try:
            self.buffer = np.append(self.buffer, audio_data)
            current_time = time.time()

            if current_time - self.last_process_time >= 5 and len(self.buffer) > 0:
                logger.info(f"Processing {len(self.buffer)} samples")
                
                timestamp_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp_str}_{mode}.wav"
                filepath = os.path.join(self.voice_dir, filename)

                with wave.open(filepath, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(44100)
                    audio_16bit = (self.buffer * 32767).astype(np.int16)
                    wav_file.writeframes(audio_16bit.tobytes())

                logger.info(f"Saved audio file: {filename}")

                future = asyncio.run_coroutine_threadsafe(
                    self.process_with_gemini(filepath), 
                    self.loop
                )
                transcription = future.result()
                
                if transcription:
                    socketio.emit('transcription', {
                        'text': transcription,
                        'timestamp': timestamp,
                        'mode': mode
                    })
                    logger.info(f"Transcription: {transcription}")

                self.buffer = np.array([], dtype=np.float32)
                self.last_process_time = current_time

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            self.buffer = np.array([], dtype=np.float32)

    def start_processing(self):
        """Start the audio processing thread"""
        self.is_running = True
        def run_event_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        threading.Thread(target=run_event_loop, daemon=True).start()
        self.processing_thread = threading.Thread(target=self.process_audio_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Audio processing thread started")

    def stop_processing(self):
        """Stop the audio processing thread"""
        self.is_running = False
        self.audio_queue.put(None)  # Signal to stop
        if self.processing_thread:
            self.processing_thread.join()
        self.loop.call_soon_threadsafe(self.loop.stop)
        logger.info("Audio processing thread stopped")

    def process_audio_queue(self):
        """Main audio processing loop"""
        buffer = np.array([], dtype=np.float32)
        last_process_time = time.time()

        while self.is_running:
            try:
                try:
                    audio_data = self.audio_queue.get(timeout=5.0)
                    if audio_data is None:
                        break
                    
                    buffer = np.append(buffer, audio_data)
                    current_time = time.time()
                    
                    # Process buffer when it's large enough (5 seconds of audio)
                    if len(buffer) >= 220500:  # 5 seconds at 44.1kHz
                        self.process_audio_chunk(buffer, int(current_time * 1000), 'both')
                        buffer = np.array([], dtype=np.float32)
                        last_process_time = current_time
                        
                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"Error in process_audio_queue: {e}", exc_info=True)
                buffer = np.array([], dtype=np.float32)

# Create audio processor instance
audio_processor = AudioProcessor()


@app.route('/voice/<path:filename>')
def serve_voice_files(filename):
    return send_from_directory('voice', filename)



@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    if not audio_processor.is_running:
        audio_processor.start_processing()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')
    audio_processor.stop_processing()

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data"""
    try:
        audio_array = np.frombuffer(data['buffer'], dtype=np.float32)
        timestamp = data.get('timestamp', int(time.time() * 1000))
        mode = data.get('mode', 'both')
        
        if len(audio_array) > 0:
            logger.debug(f"Received audio chunk - Length: {len(audio_array)}, Mode: {mode}")
            audio_processor.process_audio_chunk(audio_array, timestamp, mode)
    except Exception as e:
        logger.error(f"Error handling audio data: {e}", exc_info=True)

if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        audio_processor.stop_processing()
