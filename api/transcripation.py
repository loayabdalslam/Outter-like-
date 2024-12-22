from flask import Blueprint, request, jsonify
import os
import subprocess
import logging
from datetime import datetime
import google.generativeai as genai
import time

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Blueprint
transcription_bp = Blueprint('transcription', __name__)

# Configure Gemini
genai.configure(api_key='AIzaSyD3DIAlu69Amj0o6UKm3fhORJ3HGOdAEik', transport='rest')
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure directories
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def process_audio_with_gemini(audio_path):
    """Process audio file with Gemini API."""
    try:
        audio_file = genai.upload_file(path=audio_path)

        time.sleep(5)  # Wait 10 seconds before checking again

        prompt = """
        Please transcribe this audio file accurately and provide a clear, well-formatted transcription.
        Rules:
        1. Maintain natural speech patterns and include proper punctuation
        2. If there are multiple speakers, indicate speaker changes
        3. Include any relevant context or background sounds in [brackets]
        4. Preserve the original language of the speech
        5. If parts are unclear, indicate with [unclear] rather than guessing
        """

        response = model.generate_content(
            [prompt, audio_file],
            generation_config={
                'temperature': 0.1,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 4096,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini processing error: {e}")
        raise

def process_summary_with_gemini(text):
    """Generate summary using Gemini API."""
    try:
        prompt = f"""
        Please provide a concise summary of the following transcription:
        {text}

        Guidelines:
        1. Focus on key points and main ideas
        2. Maintain context and speaker relationships
        3. Keep the summary clear and well-structured
        4. Preserve important quotes or specific details
        5. Include timestamps or sequence of events if relevant
        """

        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.8,
                'max_output_tokens': 2048,
            }
        )
        return response.text
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        raise

@transcription_bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    logger.info("Received transcription request")
    temp_file_path = None

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        logger.info(f"Processing file: {audio_file.filename} ({audio_file.content_type})")

        if not audio_file or not audio_file.filename:
            return jsonify({'error': 'Invalid file'}), 400

        # Save temporary file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_extension = audio_file.filename.split('.')[-1].lower()
        temp_file_path = os.path.join(UPLOAD_DIR, f"audio_{timestamp}.{file_extension}")
        
        audio_file.save(temp_file_path)
        logger.info(f"Saved file: {temp_file_path}")

        # Process with Gemini
        transcription = process_audio_with_gemini(temp_file_path)
        
        if not transcription:
            raise ValueError("Empty transcription received")

        return jsonify({
            'text': transcription,
            'success': True
        })

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return jsonify({
            'error': 'Transcription failed',
            'details': str(e)
        }), 500

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

@transcription_bp.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        if not isinstance(text, str) or len(text.strip()) < 10:
            return jsonify({'error': 'Invalid text content'}), 400

        summary = process_summary_with_gemini(text)
        
        if not summary:
            raise ValueError("Empty summary received")

        return jsonify({
            'summary': summary,
            'success': True
        })

    except Exception as e:
        logger.error(f"Summary error: {e}", exc_info=True)
        return jsonify({
            'error': 'Summary generation failed',
            'details': str(e)
        }), 500


print("Starting server...")
