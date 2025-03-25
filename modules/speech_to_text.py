import openai
from pathlib import Path
from utils.audio_processor import AudioProcessor
from utils.file_manager import save_file
from config import config
import os

class SpeechToText:
    def __init__(self):
        self.input_path = config.INPUT_AUDIO
        self.output_files = {
            'text': config.OUTPUT_DIR / "transcription.txt",
            'excitement': config.OUTPUT_DIR / "excitement.csv"
        }

    def _transcribe_audio(self):
        """Core audio transcription method"""
        try:
            with open(str(self.input_path), "rb") as audio_file:
                client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                return client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                    language='en'
                )
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            raise

    def run(self):
        """Main execution method"""
        try:
            # Transcribe audio
            transcription = self._transcribe_audio()
            save_file(self.output_files['text'], transcription)
            
            # Analyze audio excitement
            excitement_data = AudioProcessor.analyze_excitement(self.input_path)
            save_file(self.output_files['excitement'], excitement_data.to_csv(index=False))
            
            return True
        except Exception as e:
            print(f"Error in speech-to-text: {str(e)}")
            return False