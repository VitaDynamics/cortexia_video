import argparse
import os
import sys

from PIL import Image

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cortexia_video.config_manager import ConfigManager
from cortexia_video.features.feature_extraction.models import CLIPFeatureExtractor


def test_image_features(image_path=None):
    """Test extracting features from images"""
    print("Testing image feature extraction...")

    # Create a config manager with the light mode config
    config_manager = ConfigManager(
        config_file_path=os.path.join("config", "light_mode.toml")
    )
    config_manager.load_config()

    # Update the config to set the required parameters
    config_manager.set_param("model_settings.clip_feature_model", "PE-Core-B16-224")
    config_manager.set_param("model_settings.clip_feature_model_identifier", "clip_pe")

    # Create a CLIP feature extractor
    feature_extractor = CLIPFeatureExtractor(config_manager)

    # Check if image path exists
    if image_path is None or not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        # Use the default input video as fallback
        video_path = config_manager.get_param("processing.input_video_path")
        if os.path.exists(video_path):
            print(f"Using first frame from video: {video_path}")
            import decord

            vr = decord.VideoReader(video_path)
            frame = vr[0].asnumpy()
            import numpy as np

            image = Image.fromarray(np.uint8(frame))
        else:
            print("No sample image or video found. Cannot proceed with testing.")
            return None
    else:
        print(f"Loading image: {image_path}")
        image = Image.open(image_path)

    # Extract features
    features = feature_extractor.extract_image_features([image])

    # Print feature shape
    print(f"Image features shape: {features.shape}")
    print(f"Feature vector norm: {features.norm().item()}")

    return features


def test_text_features():
    """Test extracting features from text"""
    print("\nTesting text feature extraction...")

    # Create a config manager with the light mode config
    config_manager = ConfigManager(
        config_file_path=os.path.join("config", "light_mode.toml")
    )
    config_manager.load_config()

    # Update the config
    config_manager.set_param("model_settings.clip_feature_model", "PE-Core-B16-224")
    config_manager.set_param("model_settings.clip_feature_model_identifier", "clip_pe")

    # Create a CLIP feature extractor
    feature_extractor = CLIPFeatureExtractor(config_manager)

    # Sample text prompts
    text_prompts = [
        "a dog running in a field",
        "a cat sitting on a windowsill",
        "a bird flying in the sky",
    ]

    # Extract features
    features = feature_extractor.extract_text_features(text_prompts)

    # Print feature shape
    print(f"Text features shape: {features.shape}")
    print(f"Feature vector norm: {features.norm(dim=-1)}")

    return features


def test_video_features(video_path=None):
    """Test extracting features from a video"""
    print("\nTesting video feature extraction...")

    # Create a config manager with the light mode config
    config_manager = ConfigManager(
        config_file_path=os.path.join("config", "light_mode.toml")
    )
    config_manager.load_config()

    # Update the config
    config_manager.set_param("model_settings.clip_feature_model", "PE-Core-B16-224")
    config_manager.set_param("model_settings.clip_feature_model_identifier", "clip_pe")

    # Create a CLIP feature extractor
    feature_extractor = CLIPFeatureExtractor(config_manager)

    # Use provided video path or get from config
    if video_path is None:
        video_path = config_manager.get_param("processing.input_video_path")

    # Check if the video exists
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        return None

    print(f"Loading video: {video_path}")

    # Extract features
    features = feature_extractor.extract_video_features(video_path, num_frames=8)

    # Print feature shape
    print(f"Video features shape: {features.shape}")
    print(f"Feature vector norm: {features.norm().item()}")

    return features


def test_similarity(image_path=None):
    """Test calculating similarity between image and text features"""
    print("\nTesting similarity calculation...")

    # Create a config manager with the light mode config
    config_manager = ConfigManager(
        config_file_path=os.path.join("config", "light_mode.toml")
    )
    config_manager.load_config()

    # Update the config
    config_manager.set_param("model_settings.clip_feature_model", "PE-Core-B16-224")
    config_manager.set_param("model_settings.clip_feature_model_identifier", "clip_pe")

    # Create a CLIP feature extractor
    feature_extractor = CLIPFeatureExtractor(config_manager)

    # Check if image path exists
    if image_path is None or not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        # Use the default input video as fallback
        video_path = config_manager.get_param("processing.input_video_path")
        if os.path.exists(video_path):
            print(f"Using first frame from video: {video_path}")
            import decord

            vr = decord.VideoReader(video_path)
            frame = vr[0].asnumpy()
            import numpy as np

            image = Image.fromarray(np.uint8(frame))
        else:
            print("No sample image or video found. Cannot proceed with testing.")
            return None
    else:
        print(f"Loading image: {image_path}")
        image = Image.open(image_path)

    # Sample text prompts
    text_prompts = [
        "a dog running in a field",
        "a cat sitting on a windowsill",
        "a bird flying in the sky",
        "a person walking on a street",
    ]

    # Extract features
    image_features = feature_extractor.extract_image_features([image])
    text_features = feature_extractor.extract_text_features(text_prompts)

    # Calculate similarity
    similarity = feature_extractor.calculate_similarity(image_features, text_features)

    # Print similarity scores
    print(f"Similarity shape: {similarity.shape}")
    for i, prompt in enumerate(text_prompts):
        print(f"Similarity with '{prompt}': {similarity[i]:.4f}")

    # Get the most similar text prompt
    most_similar_idx = similarity.argmax()
    print(f"Most similar text prompt: '{text_prompts[most_similar_idx]}'")

    return similarity


def main():
    """Run all tests"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test CLIP Feature Extractor")
    parser.add_argument(
        "--image",
        default="recognize-anything/images/openset_example.jpg",
        type=str,
        help="Path to test image",
    )
    parser.add_argument(
        "--video", default=None, type=str, help="Path to test video (optional)"
    )

    # Parse arguments
    args = parser.parse_args()

    print("Starting CLIP Feature Extractor tests...")

    # Test image feature extraction
    image_features = test_image_features(args.image)

    # Test text feature extraction
    text_features = test_text_features()

    # Test video feature extraction
    video_features = test_video_features(args.video)

    # Test similarity calculation
    similarity = test_similarity(args.image)

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
