import cv2
import numpy as np
import random

class VideoEffects:
    @staticmethod
    def apply_effects(frame, excitement_data):
        """Apply dynamic effects based on excitement"""
        # Base effects
        frame = cv2.resize(frame, config.VIDEO_SIZE)
        
        # Add background effects
        bg = np.zeros((*config.VIDEO_SIZE[::-1], 3), dtype=np.uint8)
        bg[:,:] = VideoEffects._get_bg_color(excitement_data)
        
        # Composite frame
        x = (config.VIDEO_SIZE[0] - frame.shape[1]) // 2
        y = (config.VIDEO_SIZE[1] - frame.shape[0]) // 2
        bg[y:y+frame.shape[0], x:x+frame.shape[1]] = frame
        
        return bg

    @staticmethod
    def _get_bg_color(excitement):
        """Generate dynamic background color"""
        hue = int(210 - (excitement * 60))  # Blue to purple
        saturation = int(70 + (excitement * 30))
        value = int(80 - (excitement * 30))
        hsv = np.uint8([[[hue, saturation, value]]])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]