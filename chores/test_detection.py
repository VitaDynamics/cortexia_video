#!/usr/bin/env python
import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from cortexia.config_manager import ConfigManager
from cortexia.features.detection.models import ObjectDetector


def visualize_detections(image, detections, output_path=None):
    """Visualize detections on the image."""
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))

    # Display the image
    ax.imshow(image)

    # Different colors for different objects
    colors = ["r", "g", "b", "c", "m", "y", "k"]

    # Draw each detection
    for i, detection in enumerate(detections):
        color = colors[i % len(colors)]
        box = detection["box"]
        label = detection["label"]
        score = detection["score"]

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add label
        plt.text(
            box[0],
            box[1] - 5,
            f"{label}: {score:.2f}",
            color=color,
            fontsize=12,
            backgroundcolor="white",
        )

    # Remove axis ticks
    plt.axis("off")

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Detection visualization saved to {output_path}")

    plt.show()


def test_object_detector(
    image_path, text_prompts=None, confidence=0.5, output_path=None
):
    """Test the ObjectDetector with a sample image."""
    # Default text prompts if none provided
    if text_prompts is None:
        text_prompts = ["person", "car", "dog", "cat", "bicycle"]

    # Create and setup config manager
    config_manager = ConfigManager()
    config_manager.set_param(
        "model_settings.object_detection_model", "IDEA-Research/grounding-dino-tiny"
    )

    # Initialize the ObjectDetector
    detector = ObjectDetector(config_manager)

    # Check if image path exists
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist")
        return

    # Load the image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)

    # Convert text prompts to list format for batch processing
    batch_prompts = [text_prompts] * 3  # For a single image
    images_batch = [image] * 3  # Batch with a single image

    # Detect objects in the image
    print(f"Detecting objects with prompts: {text_prompts}")
    detections_batch = detector.detect_objects(
        images_batch, batch_prompts
    )

    # Get detections for our single image
    detections = detections_batch[0]

    # Print results
    print("\nDetected objects:")
    for i, detection in enumerate(detections):
        print(f"Detection {i + 1}:")
        print(f"  Label: {detection['label']}")
        print(f"  Score: {detection['score']:.4f}")
        print(f"  Box: {detection['box']}")

    # Visualize if requested
    if output_path or len(detections) > 0:
        visualize_detections(image, detections, output_path)

    return detections


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test ObjectDetector")
    parser.add_argument("--image", default="recognize-anything/images/openset_example.jpg", type=str, help="Path to test image")
    parser.add_argument(
        "--prompts",
        type=str,
        default="person,car,dog,cat",
        help="Comma-separated text prompts for object detection",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save visualization output"
    )

    # Parse arguments
    args = parser.parse_args()

    # Process prompts
    prompts = [p.strip() for p in args.prompts.split(",")]

    # Run the test
    test_object_detector(args.image, prompts, args.confidence, args.output)
