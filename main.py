import sys
import time
from tqdm import tqdm
from modules.speech_to_text import SpeechToText
from modules.text_to_gloss import TextToGloss
from modules.video_generator import VideoGenerator
from modules.skeleton_overlay import SkeletonOverlay
from config import config

def run_pipeline():
    """Execute the complete sign avatar pipeline"""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    steps = [
        ("Audio to Text", SpeechToText().run),
        ("Text to Gloss", TextToGloss().run),
        ("Generate Video", VideoGenerator().run),
        ("Add Skeleton", SkeletonOverlay().run)
    ]
    
    for step_name, step_func in tqdm(steps, desc="Processing Pipeline"):
        print(f"\n{'='*50}\n{step_name}\n{'='*50}")
        if not step_func():
            print(f"Pipeline failed at: {step_name}")
            return False
    
    print("\nPipeline completed successfully!")
    return True

if __name__ == "__main__":
    start_time = time.time()
    
    if run_pipeline():
        final_output = config.OUTPUT_DIR / "final_with_skeleton.mp4"
        print(f"\nFinal output created at: {final_output}")
        print(f"Total processing time: {time.time()-start_time:.2f} seconds")