#%%
from flask import Flask,send_from_directory,jsonify
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
from pyannote.audio import Pipeline
from transformers import BertTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoTokenizer
from arabert.preprocess import ArabertPreprocessor

load_dotenv()

DIARIZATION_PIPELINE = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token='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'
)


genai.configure(api_key="AIzaSyA-Ayj7gvkSjdFdwlFBgUcRWTz9lPYEb5U")
model = genai.GenerativeModel('gemini-1.5-flash')
#%%
class TextSummarizer:
    def __init__(self, model_name="malmarjeh/mbert2mbert-arabic-text-summarization"):
        self.model_name = model_name
        self.preprocessor = ArabertPreprocessor(model_name="")
        if 't5' not in model_name:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def summarize_text(self, text, max_length=200, num_beams=3, length_penalty=1.0, repetition_penalty=3.0, no_repeat_ngram_size=3):
        prep_text = self.preprocessor.preprocess(text)

        summary = self.pipeline(
            prep_text,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size)[0]['generated_text']

        return summary

summarizer = TextSummarizer()
#%%
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
        if not os.path.isfile(audio_file_path):
            raise ValueError(f"Invalid audio file path: {audio_file_path}")

        try:
            audio_file = genai.upload_file(path=audio_file_path)
            prompt = f"""
            Please transcribe this audio file and provide a clear, well-formatted transcription.
            The audio is a WAV file containing speech that needs to be transcribed accurately.
            Please maintain natural speech patterns and include proper punctuation, known that
            the audio main language used is Arabic, with a few minority of English words
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


    def extract_notes_with_ai(self,transcription):
        try:
            prompt = f"""
            The following is a meeting transcription. Please extract any notes or important observations, words like
            note, observation, important, take care of,
            ملاحظة, ملحوظة, مهم, لاحظ انه, خلي بالك, دير بالك, خلي عندنا, اكتب عندك , سجل الحتة ديه
            and anything similar, just gimme the notes like bullets don't describe anythinh don't tell me
            that you didn't have any notes if you haven't any just say no notes, don't describe anything
            at all or add any comments just notes each in a line and that's it now you have the transcription:
            {transcription}
            Notes (in Arabic):
            """
            response = model.generate_content([prompt])
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error extracting notes with AI: {e}", exc_info=True)
            return "تعذر استخراج الملاحظات"  # Return an appropriate error in Arabic

    def diarize_audio(self, audio_file_path: str):
        diarization_result = DIARIZATION_PIPELINE(audio_file_path)
        speaker_segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        return speaker_segments

    def split_audio(self, audio_file_path: str, segments: list):
        """Split audio file into chunks based on diarization segments."""
        with wave.open(audio_file_path, 'rb') as wav:
            frame_rate = wav.getframerate()
            num_channels = wav.getnchannels()
            samp_width = wav.getsampwidth()

            audio_chunks = []
            for segment in segments:
                start_frame = int(segment["start"] * frame_rate)
                end_frame = int(segment["end"] * frame_rate)
                wav.setpos(start_frame)
                frames = wav.readframes(end_frame - start_frame)

                chunk_path = os.path.join(VOICE_DIR, f"{segment['speaker']}_{segment['start']:.2f}_{segment['end']:.2f}.wav")
                with wave.open(chunk_path, 'wb') as chunk_wav:
                    chunk_wav.setnchannels(num_channels)
                    chunk_wav.setsampwidth(samp_width)
                    chunk_wav.setframerate(frame_rate)
                    chunk_wav.writeframes(frames)

                audio_chunks.append({
                    "file_path": chunk_path,
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"]
                })

            return audio_chunks

    async def transcribe_and_diarize(self, audio_file_path: str):
        """Perform transcription and diarization."""
        # Step 1: Perform diarization
        diarization = self.diarize_audio(audio_file_path)

        # Step 2: Split audio into chunks
        audio_chunks = self.split_audio(audio_file_path, diarization)

        # Step 3: Transcribe each chunk
        speaker_transcriptions = []
        for chunk in audio_chunks:
            transcription = await self.transcribe_audio_chunk(chunk["file_path"])
            speaker_transcriptions.append({
                "speaker": chunk["speaker"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": transcription
            })

        return speaker_transcriptions

audio_processor = AudioProcessor()
#%%
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120)

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

@app.route('/transcribe/<filename>', methods=['GET'])
def transcribe_audio_file(filename):
    filepath = os.path.join(VOICE_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        transcription = asyncio.run(audio_processor.process_with_gemini(filepath))

        if transcription:
            return transcription, 200, {'Content-Type': 'text/plain; charset=utf-8'}

        else:
            return jsonify({"error": "Failed to transcribe the audio file"}), 500
    except Exception as e:
        logger.error(f"Error transcribing audio file: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during transcription"}), 500

@app.route('/extract_notes/<filename>', methods=['GET'])
def extract_notes(filename):
    filepath = os.path.join(VOICE_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        transcription = asyncio.run(audio_processor.process_with_gemini(filepath))
        if transcription:
            notes = audio_processor.extract_notes_with_ai(transcription)
            return notes, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        else:
            return jsonify({"error": "Failed to transcribe the audio file"}), 500
    except Exception as e:
        logger.error(f"Error extracting notes: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during note extraction"}), 500

@app.route('/find_summary/<filename>', methods=['GET'])
def find_summary(filename):
    filepath = os.path.join(VOICE_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        transcription = asyncio.run(audio_processor.process_with_gemini(filepath))
        if transcription:
            summary = summarizer.summarize_text(transcription)
            return summary, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        else:
            return jsonify({"error": "Failed to transcribe the audio file"}), 500
    except Exception as e:
        logger.error(f"Error extracting notes: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during note extraction"}), 500

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