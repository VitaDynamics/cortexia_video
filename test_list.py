import argparse
import os

from PIL import Image

from cortexia_video.config_manager import ConfigManager
from cortexia_video.object_listing import RAMLister


def test_ram_lister(image_path=None):
    """Test the RAMLister with a sample image."""
    # Create a config manager with default settings
    config_manager = ConfigManager()
    config_manager.set_param(
        "model_settings.object_listing_model", "recognize_anything/ram"
    )

    # Initialize the RAMLister
    lister = RAMLister(config_manager)

    # Use default image if not provided
    if image_path is None:
        raise ValueError("Image path is required")
    # Check if image path exists
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist")
        return

    # Load the image and process it
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)

    # List objects in the image
    print("Detecting objects...")
    objects = lister.list_objects_in_image(image)

    # Print results
    print("\nDetected objects:")
    for i, obj in enumerate(objects, 1):
        print(f"{i}. {obj}")

    return objects


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test RAMLister object detection")
    parser.add_argument("--image", type=str, help="Path to test image (optional)")

    # Parse arguments
    args = parser.parse_args()

    # Run the test
    test_ram_lister(args.image)
