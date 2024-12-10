#%%
from flask import Flask, send_from_directory
from flask_cors import CORS
import logging
from flask_socketio import SocketIO
import numpy as np
import wave
import queue
import time
import os
import asyncio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import csv
from datetime import datetime
#%%
TIMING_CSV = "processing_times.csv"
if not os.path.exists(TIMING_CSV):
    with open(TIMING_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Chunk Duration (sec)", "Processing Time (sec)"])
#%%
# Initialize Hugging Face model
processor = AutoProcessor.from_pretrained("MohamedRashad/Arabic-Whisper-CodeSwitching-Edition")
model = AutoModelForSpeechSeq2Seq.from_pretrained("MohamedRashad/Arabic-Whisper-CodeSwitching-Edition")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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

    def stop_processing(self):
        """Stop the audio processing thread"""
        self.is_running = False
        if self.processing_thread is not None:
            self.processing_thread.join()
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        logger.info("Audio processing thread stopped")

    async def process_with_huggingface(self, audio_file_path: str, chunk_duration: int = None) -> str:
        """Process audio file with Hugging Face model, log timing."""
        if not os.path.isfile(audio_file_path):
            raise ValueError(f"Invalid audio file path: {audio_file_path}")

        try:
            start_time = time.time()  # Start timing

            # Load and process the audio file
            audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

            # Resample audio data to 16000 Hz if it's not already
            if sample_rate != 16000:
                logger.info(f"Resampling audio from {sample_rate} Hz to 16000 Hz")
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

            # Prepare the input for the model
            inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")

            # Generate transcription with the model
            with torch.no_grad():
                generated_ids = model.generate(inputs["input_features"])

            # Decode the generated ids to text
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
            result = " ".join(transcription)

            # Measure processing time
            end_time = time.time()
            processing_time = end_time - start_time

            # Log timing to CSV
            with open(TIMING_CSV, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().isoformat(), chunk_duration, processing_time])

            logger.info(f"Processing Time: {processing_time:.2f} seconds for duration: {chunk_duration} sec")
            logger.info(f"Transcription result: {result}")

            return result
        except Exception as e:
            logger.exception(f"Unexpected error processing audio: {e}")
            return ""

    def process_audio_chunk(self, audio_data: np.ndarray, timestamp: int, mode: str):
        try:
            self.buffer = np.append(self.buffer, audio_data)
            chunk_samples = 44100 * 3  # 3 seconds worth of audio at 44.1kHz

            while len(self.buffer) >= chunk_samples:
                # Extract a chunk of audio data
                chunk = self.buffer[:chunk_samples]
                self.buffer = self.buffer[chunk_samples:]

                logger.info(f"Processing chunk of {len(chunk)} samples")

                # Save the chunk to a temporary file
                timestamp_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp_str}_{mode}.wav"
                filepath = os.path.join(self.voice_dir, filename)

                with wave.open(filepath, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(44100)
                    audio_16bit = (chunk * 32767).astype(np.int16)
                    wav_file.writeframes(audio_16bit.tobytes())

                logger.info(f"Saved audio chunk: {filename}")

                # Transcribe the chunk
                future = asyncio.run_coroutine_threadsafe(
                    self.process_with_huggingface(filepath),
                    self.loop
                )
                transcription = future.result()

                # Emit the transcription
                if transcription:
                    socketio.emit('transcription', {
                        'text': transcription,
                        'timestamp': timestamp,
                        'mode': mode
                    })
                    logger.info(f"Chunk Transcription: {transcription}")

            # Update the last process time to throttle processing
            self.last_process_time = time.time()

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            self.buffer = np.array([], dtype=np.float32)


# Create audio processor instance
audio_processor = AudioProcessor()
#%%
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

@app.route('/voice/<path:filename>')
def serve_voice_files(filename):
    return send_from_directory('voice', filename)

@app.route('/transcribe/<filename>')
def transcribe_file(filename):
    file_path = os.path.join(VOICE_DIR, filename)
    if not os.path.isfile(file_path):
        return {"error": "File not found"}, 404
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        transcription = loop.run_until_complete(audio_processor.process_with_huggingface(file_path))
        loop.close()
        return transcription, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error in transcription: {e}", exc_info=True)
        return {"error": str(e)}, 500

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
#%%
if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
    finally:
        audio_processor.stop_processing()
        print('Processing is done')