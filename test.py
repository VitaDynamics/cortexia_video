from cortexia_video.object_detection import ObjectDetector
from PIL import Image
import cv2
import numpy as np
from cortexia_video.config_manager import ConfigManager

config = ConfigManager("config/", "config")
config.load_config()
detector = ObjectDetector(config)

try:
    print("Loading video frame...")
    video = cv2.VideoCapture("sample_data/input.mp4")
    ret, frame = video.read()
    if not ret:
        print("Failed to read frame from video")
        exit(1)
    
    # Convert BGR to RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    print("Frame loaded successfully")
except Exception as e:
    print(f"Failed to load video frame: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    print("Detecting objects...")
    detections = detector.detect_objects(image, "person. car. dog. cat.")
    print(f"Found {len(detections)} detections:")
    for i, det in enumerate(detections):
        print(f"Detection {i+1}:")
        print(f"  Label: {det['label']}")
        print(f"  Score: {det['score']:.4f}")
        print(f"  Box: {det['box']}")
except Exception as e:
    print(f"Error during detection: {e}")
    import traceback
    traceback.print_exc() 