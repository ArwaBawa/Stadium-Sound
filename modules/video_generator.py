import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import Config

class VideoGenerator:
    def __init__(self):
        self.sign_dataset = self._load_annotations()
        self.current_time = 0.0
        self.background = self._create_scrolling_gradient()
        self.transition_frames = self._create_transition_effect()

    def _load_annotations(self):
        """Load sign annotations with robust error handling"""
        try:
            if Config.ANNOTATIONS_FILE.exists():
                return pd.read_csv(Config.ANNOTATIONS_FILE)
            print(f"⚠️ Annotations file not found at {Config.ANNOTATIONS_FILE}")
            return pd.DataFrame(columns=['label', 'video_filename'])
        except Exception as e:
            print(f"⚠️ Error loading annotations: {e}")
            return pd.DataFrame(columns=['label', 'video_filename'])

    def _create_scrolling_gradient(self):
        """Create a horizontally scrolling gradient background"""
        width, height = Config.VIDEO_SIZE
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create HSV gradient (better color control)
        for x in range(width):
            hue = int(180 * (x / width))  # 0-180 range for OpenCV
            gradient[:, x] = [hue, 255, 255]
        
        return cv2.cvtColor(gradient, cv2.COLOR_HSV2BGR)

    def _create_transition_effect(self):
        """Generate smooth transition frames between signs"""
        width, height = Config.VIDEO_SIZE
        frames = []
        for alpha in np.linspace(0, 1, Config.FPS//2):  # Half-second transition
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(
                frame,
                (width//2, height//2),
                int(height*alpha),
                (255, 255, 255),
                -1
            )
            frames.append(frame)
        return frames

    def _find_sign_video(self, word):
        """Hierarchical video path resolution"""
        word_clean = word.strip().lower()
        
        # 1. Check in annotations
        if not self.sign_dataset.empty:
            match = self.sign_dataset[
                self.sign_dataset['label'].str.lower() == word_clean
            ]
            if not match.empty:
                path = Config.SIGN_DATASET / match.iloc[0]['video_filename']
                if path.exists():
                    return path
        
        # 2. Check fingerspelling
        single_letter_path = Config.FINGERSPELLING / f"{word_clean.upper()}.mp4"
        if single_letter_path.exists():
            return single_letter_path
        
        # 3. Spell multi-letter words
        if len(word_clean) > 1:
            return self._spell_word_fingerspelling(word_clean)
        
        return None

    def _spell_word_fingerspelling(self, word):
        """Generate letter-by-letter paths with validation"""
        paths = []
        for letter in word:
            letter_path = Config.FINGERSPELLING / f"{letter.upper()}.mp4"
            if letter_path.exists():
                paths.append(letter_path)
            else:
                print(f"⚠️ Missing fingerspelling for '{letter}' in '{word}'")
        return paths if paths else None

    def _process_sign_frame(self, frame):
        """Composite sign video frame with background"""
        # Resize maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w, target_h = Config.VIDEO_SIZE
        
        # Calculate padding
        scale = min(target_w/w, target_h/h)
        new_size = (int(w*scale), int(h*scale))
        frame = cv2.resize(frame, new_size)
        
        # Create centered composition
        composite = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_size[0]) // 2
        y_offset = (target_h - new_size[1]) // 2
        composite[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = frame
        
        # Create mask from non-black pixels
        gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Animate background
        bg_offset = int(self.current_time * 15) % Config.VIDEO_SIZE[0]
        bg = np.roll(self.background, bg_offset, axis=1)
        
        # Composite layers
        foreground = cv2.bitwise_and(composite, composite, mask=mask)
        background = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
        return cv2.add(foreground, background)

    def _add_transition(self, out, duration=0.5):
        """Insert transition animation between signs"""
        for frame in self.transition_frames[:int(Config.FPS*duration)]:
            out.write(frame)
            self.current_time += 1/Config.FPS

    def run(self):
        """Main video generation workflow"""
        try:
            # Setup output
            Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = Config.OUTPUT_DIR / "signed_output.mp4"
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                Config.FPS,
                Config.VIDEO_SIZE
            )
            
            # Load gloss text
            gloss_path = Config.OUTPUT_DIR / "gloss.txt"
            if not gloss_path.exists():
                raise FileNotFoundError(f"Gloss file missing: {gloss_path}")
                
            with open(gloss_path) as f:
                gloss_words = [w.strip() for w in f.read().split() if w.strip()]
            
            # Process each sign
            for i, word in enumerate(tqdm(gloss_words, desc="Generating Sign Video")):
                video_source = self._find_sign_video(word)
                if not video_source:
                    print(f"⏩ Skipping: '{word}' (no matching sign)")
                    continue
                
                # Add transition between signs (except first)
                if i > 0:
                    self._add_transition(out)
                
                # Handle both single videos and spelling sequences
                sources = [video_source] if isinstance(video_source, Path) else video_source
                for path in sources:
                    cap = cv2.VideoCapture(str(path))
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        out.write(self._process_sign_frame(frame))
                        self.current_time += 1/Config.FPS
                    cap.release()
            
            out.release()
            print(f"✅ Successfully created: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return False