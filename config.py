
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    ASSETS_DIR = BASE_DIR / "assets"
    
    # Path configurations
    INPUT_AUDIO = DATA_DIR / "input" / "audio.m4a"
    OUTPUT_DIR = DATA_DIR / "output"
    ANNOTATIONS_FILE = DATA_DIR / "annotations.csv"
    SIGN_DATASET = ASSETS_DIR / "sign_dataset"
    FINGERSPELLING = ASSETS_DIR / "Fingerspelling"

    OPENAI_API_KEY = "add your"

    # Processing settings
    VIDEO_SIZE = (640, 480)
    FPS = 30
    KEEP_INTERMEDIATES = False
        # Color settings (ADD THESE NEW LINES)
    BACKGROUND_SATURATION = 90  # 0-100% (higher = more vibrant)
    BACKGROUND_LIGHTNESS = 50   # 0-100% (50 = normal, lower = darker)
    BACKGROUND_HUE_RANGE = (120, 0)  # Blue (120°) to Red (0°) - you can adjust these values
    SKELETON_COLOR = (0, 255, 0)  # Green lines
    SKELETON_THICKNESS = 3  # Thicker lines

config = Config()
