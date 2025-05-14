import argparse

from PIL import Image

from cortexia_video.object_listing import OBJECT_LISTER_REGISTRY


def test_object_listing(image_path, model_name):
    """Test the object listing functionality with a given image and model"""

    # Create a simple config manager that returns the model name
    class SimpleConfigManager:
        def get_param(self, param_name):
            if param_name == "model_settings.object_listing_model":
                return model_name
            return None

    config_manager = SimpleConfigManager()

    # Get the appropriate lister class from the registry
    if model_name not in OBJECT_LISTER_REGISTRY:
        print(f"Error: Model {model_name} not found in registry.")
        print(f"Available models: {list(OBJECT_LISTER_REGISTRY.keys())}")
        return

    # Create an instance of the lister
    lister_class = OBJECT_LISTER_REGISTRY[model_name]
    lister = lister_class(config_manager)

    # Load the image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Run object listing
    print(f"Running object listing with model: {model_name}")
    objects = lister.list_objects_in_image(image)

    # Print results
    print("\nDetected objects:")
    for i, obj in enumerate(objects, 1):
        print(f"{i}. {obj}")

    print(f"\nTotal objects detected: {len(objects)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test object listing functionality")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="recognize_anything/ram",
        choices=list(OBJECT_LISTER_REGISTRY.keys()),
        help="Model to use for object listing",
    )

    args = parser.parse_args()

    test_object_listing(args.image, args.model)
