import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from config import config
from tqdm import tqdm
import pandas as pd

class SkeletonOverlay:
    def __init__(self):
        # Initialize MediaPipe with custom drawing specs
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        # Custom drawing specs (no dots, only lines)
        self.connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=config.SKELETON_THICKNESS,
            color=config.SKELETON_COLOR
        )
        # Empty spec for landmarks (invisible)
        self.landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=0,
            circle_radius=0
        )

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _create_background(self, excitement):
        """Create dynamic colored background"""
        hue = int(240 * (1 - excitement))  # Blue to red
        hls = np.uint8([[[hue, config.BACKGROUND_LIGHTNESS, config.BACKGROUND_SATURATION]]])
        return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)[0][0]

    def _add_effects(self, image, excitement):
        """Add sparkle effects during high excitement"""
        if excitement > 0.7:
            h, w = image.shape[:2]
            for _ in range(int(50 * excitement)):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                cv2.circle(image, (x, y), 
                          np.random.randint(1, 3), 
                          (255, 255, 255), -1)

    def _draw_clean_skeleton(self, image, results_pose, results_hands):
        """Draw only skeleton connections (no dots)"""
        # Draw pose connections
        if results_pose.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                results_pose.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_drawing_spec,
                connection_drawing_spec=self.connection_drawing_spec)
        
        # Draw hand connections
        if results_hands.multi_hand_landmarks:
            for landmarks in results_hands.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec,
                    connection_drawing_spec=self.connection_drawing_spec)

    def process_frame(self, frame, excitement):
        """Process single frame to create clean skeleton visualization"""
        # Create colored background
        background = np.full((*config.VIDEO_SIZE[::-1], 3), 
                           self._create_background(excitement),
                           dtype=np.uint8)
        
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(frame_rgb)
        results_hands = self.hands.process(frame_rgb)
        
        # Draw clean skeleton
        self._draw_clean_skeleton(background, results_pose, results_hands)
        
        # Add effects
        self._add_effects(background, excitement)
        
        return background

    def run(self):
        """Process entire video"""
        try:
            # Setup video I/O
            input_path = config.OUTPUT_DIR / "final_video.mp4"
            output_path = config.OUTPUT_DIR / "final_with_skeleton.mp4"
            
            cap = cv2.VideoCapture(str(input_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, config.VIDEO_SIZE)
            
            # Load excitement data
            excitement_data = pd.read_csv(config.OUTPUT_DIR / "excitement.csv")
            
            # Process frames
            for frame_num in tqdm(range(total_frames), desc="Generating Skeleton"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get current excitement
                current_time = frame_num / fps
                excitement = excitement_data[
                    excitement_data['timestamp'] <= current_time
                ].iloc[-1]['excitement']
                
                # Process and write frame
                out.write(self.process_frame(frame, excitement))
            
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            print(f"Skeleton generation error: {str(e)}")
            return False