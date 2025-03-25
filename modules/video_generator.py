import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import config

class VideoGenerator:
    def __init__(self):
        # Load datasets
        self.df = pd.read_csv(config.ANNOTATIONS_FILE)
        self.df['label'] = self.df['label'].str.strip().str.lower()
        self.excitement_data = pd.read_csv(config.OUTPUT_DIR / "excitement.csv")
        self.current_time = 0.0

    def _find_video_path(self, word):
        """Find video path for a word with fingerspelling fallback"""
        word_clean = word.strip().lower()
        
        # 1. Check in sign dataset
        match = self.df[self.df['label'] == word_clean]
        if not match.empty:
            path = config.SIGN_DATASET / match.iloc[0]['video_filename']
            if path.exists():
                return path
        
        # 2. Try fingerspelling
        if len(word_clean) == 1:  # Single letter
            path = config.FINGERSPELLING / f"{word_clean.upper()}.mp4"
            if path.exists():
                return path
        
        # 3. Full word fingerspelling fallback
        print(f"Word '{word}' not found - attempting fingerspelling...")
        return self._spell_word(word_clean)

    def _spell_word(self, word):
        """Generate paths for spelling out a word"""
        paths = []
        for letter in word:
            path = config.FINGERSPELLING / f"{letter.upper()}.mp4"
            if path.exists():
                paths.append(path)
            else:
                print(f"Missing fingerspelling video for '{letter}'")
        return paths if paths else None

    def _get_excitement_level(self):
        """Get current excitement level based on timestamp"""
        try:
            # Find the latest excitement value before current_time
            return self.excitement_data[
                self.excitement_data['timestamp'] <= self.current_time
            ].iloc[-1]['excitement']
        except:
            return 0.5  # Default if no data

    def _apply_dynamic_background(self, frame):
        """Replace background with excitement-based color"""
        # Get current excitement color
        excitement = self._get_excitement_level()
        hue = int(240 * (1 - excitement))  # Blue (240°) to Red (0°)
        hls = np.uint8([[[hue, config.BACKGROUND_LIGHTNESS, config.BACKGROUND_SATURATION]]])
        bg_color = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)[0][0]
        
        # Create background
        background = np.full((*config.VIDEO_SIZE[::-1], 3), bg_color, dtype=np.uint8)
        
        # Resize and center frame
        frame = cv2.resize(frame, (int(config.VIDEO_SIZE[0] * 0.8), 
                              int(config.VIDEO_SIZE[1] * 0.8)))
        x = (config.VIDEO_SIZE[0] - frame.shape[1]) // 2
        y = (config.VIDEO_SIZE[1] - frame.shape[0]) // 2
        
        # Composite frame over background
        background[y:y+frame.shape[0], x:x+frame.shape[1]] = frame
        
        # Add effects for high excitement
        if excitement > 0.7:
            self._add_sparkles(background, intensity=excitement)
        
        return background

    def _add_sparkles(self, image, intensity):
        """Add sparkle effects to image"""
        h, w = image.shape[:2]
        for _ in range(int(50 * intensity)):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(1, 4)
            cv2.circle(image, (x, y), radius, (255, 255, 255), -1)

    def run(self):
        """Generate the final video with dynamic backgrounds"""
        try:
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(config.OUTPUT_DIR / "final_video.mp4"),
                fourcc, config.FPS, config.VIDEO_SIZE)
            
            # Process each word
            with open(config.OUTPUT_DIR / "gloss.txt") as f:
                gloss_words = [w.strip() for w in f.read().split() if w.strip()]
            
            for word in tqdm(gloss_words, desc="Generating Video"):
                video_source = self._find_video_path(word)
                if not video_source:
                    continue
                
                # Handle both single videos and spelling sequences
                sources = [video_source] if isinstance(video_source, Path) else video_source
                for path in sources:
                    cap = cv2.VideoCapture(str(path))
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame with dynamic background
                        processed_frame = self._apply_dynamic_background(frame)
                        out.write(processed_frame)
                        
                        # Update timeline
                        self.current_time += 1/config.FPS
                    
                    cap.release()
            
            out.release()
            return True
            
        except Exception as e:
            print(f"Video generation failed: {str(e)}")
            return False