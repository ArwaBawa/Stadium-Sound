import librosa
import pandas as pd
from pathlib import Path

class AudioProcessor:
    @staticmethod
    def analyze_excitement(audio_path):
        """Robust audio analysis with proper path handling"""
        try:
            # Convert Path object to string if needed
            audio_str = str(audio_path) if isinstance(audio_path, Path) else audio_path
            
            # Load audio with more robust backend
            y, sr = librosa.load(audio_str, sr=None, res_type='kaiser_fast')
            
            # Your existing analysis code here
            return pd.DataFrame({
                'timestamp': [0.0, 1.0, 2.0],  # Replace with real analysis
                'excitement': [0.3, 0.8, 0.5]   # Replace with real values
            })
            
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            raise