import os
import cv2
import pandas as pd
from pathlib import Path
from config import config

def create_annotations():
    """Generate annotations CSV for sign language videos"""
    data = []
    video_extensions = ('.mp4', '.mov', '.MP4', '.MOV')
    
    print(f"Scanning {config.SIGN_DATASET} for videos...")
    
    # Process sign dataset
    for file in os.listdir(config.SIGN_DATASET):
        if file.endswith(video_extensions):
            try:
                video_path = config.SIGN_DATASET / file
                cap = cv2.VideoCapture(str(video_path))
                
                if not cap.isOpened():
                    print(f"Warning: Couldn't open {file}")
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = round(frames / fps, 2) if fps > 0 else 0
                cap.release()
                
                data.append({
                    'video_filename': file,
                    'label': Path(file).stem.lower(),
                    'duration': duration,
                    'fps': round(fps) if fps > 0 else 0
                })
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    # Create directory if needed
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    if data:
        pd.DataFrame(data).to_csv(config.ANNOTATIONS_FILE, index=False)
        print(f"Created annotations at {config.ANNOTATIONS_FILE}")
        return True
    else:
        print("No valid videos found!")
        return False

if __name__ == "__main__":
    create_annotations()