# modules/skeleton_overlay.py
import cv2
import numpy as np
import mediapipe as mp
import librosa
from pathlib import Path
from tqdm import tqdm
from config import Config
import subprocess

class SkeletonOverlay:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize models
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Audio visualization parameters
        self.sr = 22050  # Sample rate
        self.fps = 30    # Video frame rate
        self.lines = 6   # Number of wave lines

    def load_audio(self, audio_path):
        """Load and process audio file for visualization"""
        self.audio, _ = librosa.load(audio_path, sr=self.sr)
        self.frame_size = int(self.sr / self.fps)
        self.total_frames = int(len(self.audio) / self.frame_size)
        
        # Calculate volume levels
        self.volume_levels = librosa.feature.rms(
            y=self.audio, 
            frame_length=self.frame_size, 
            hop_length=self.frame_size
        )[0]
        # Normalize volume levels
        self.volume_levels = (self.volume_levels - self.volume_levels.min()) / \
                           (self.volume_levels.max() - self.volume_levels.min() + 1e-6)

    def generate_wave_background(self, frame_num, width, height):
        """Generate reactive wave background for current frame"""
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        
        if frame_num >= len(self.volume_levels):
            return bg
        
        volume = self.volume_levels[frame_num]
        x = np.linspace(0, width, width)
        
        for i in range(self.lines):
            # Wave parameters
            frequency = 0.002 + i * 0.0004
            amplitude = (20 + volume * 60) * (0.6 + i / self.lines)
            y_offset = i * 18 + height//3
            
            # Generate wave points
            y = np.sin(2 * np.pi * (frequency * x + frame_num * 0.02)) * amplitude + y_offset
            
            # Color based on volume
            if volume > 0.7:
                color = (255, int(100 + 155 * volume), 100)  # Warm colors
            elif volume < 0.3:
                color = (100, 100, int(100 + 155 * (1-volume)))  # Cool colors
            else:
                hue = i / self.lines
                color = (int(255 * hue), int(255 * (1 - hue)), 128)  # Rainbow
            
            # Draw wave line
            points = np.column_stack((x, y)).astype(np.int32)
            cv2.polylines(bg, [points], isClosed=False, color=color, thickness=2)
        
        return bg

    def process_frame(self, frame, frame_num):
        """Process single frame with skeleton and wave background"""
        height, width = frame.shape[:2]
        
        # Generate reactive background
        background = self.generate_wave_background(frame_num, width, height)
        
        # Detect skeleton
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(frame_rgb)
        results_hands = self.hands.process(frame_rgb)
        
        # Draw skeleton on background
        if results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                background,
                results_pose.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=Config.SKELETON_COLOR,
                    thickness=Config.SKELETON_THICKNESS)
            )
        
        if results_hands.multi_hand_landmarks:
            for landmarks in results_hands.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    background,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=Config.HAND_COLOR,
                        thickness=Config.SKELETON_THICKNESS)
                )
        
        return background

    def run(self):
        """Process video with skeleton overlay and wave background"""
        try:
            input_path = Config.OUTPUT_DIR / "signed_output.mp4"
            audio_path = "data/input/football-crowd-goal.wav"
            output_path = Config.OUTPUT_DIR / "skeleton_output.mp4"
            
            # Load audio data
            self.load_audio(str(audio_path))
            
            # Open video file
            cap = cv2.VideoCapture(str(input_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height))
            
            # Process frames with progress bar
            for frame_num in tqdm(range(total_frames), desc="Adding Skeleton & Waves"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed = self.process_frame(frame, frame_num)
                out.write(processed)
            
            # Release resources
            cap.release()
            out.release()
            
            # Combine with audio
            temp_output = output_path.with_name(f"temp_{output_path.name}")
            command = [
                "ffmpeg", "-y",
                "-i", str(output_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(temp_output)
            ]
            subprocess.run(command, check=True)
            temp_output.replace(output_path)
            
            return True
            
        except Exception as e:
            print(f"Skeleton processing error: {e}")
            return False