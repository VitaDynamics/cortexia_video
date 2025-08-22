#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cortexia_video.config_manager import ConfigManager
from cortexia_video.features.segmentation.models import ObjectSegmenter


def visualize_segmentation(
    image, mask, input_boxes=None, input_points=None, output_path=None
):
    """Visualize segmentation mask on the image."""
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))

    # Display the image
    ax.imshow(image)

    # Display the segmentation mask as a semi-transparent overlay
    mask_overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_overlay[mask > 0] = [255, 0, 0, 128]  # Red with alpha transparency
    ax.imshow(mask_overlay)

    # Draw input boxes if provided
    if input_boxes is not None:
        for box in input_boxes:
            x1, y1, x2, y2 = map(int, box)
            # Create a Rectangle patch
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="white",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)

    # Draw input points if provided
    if input_points is not None:
        for points in input_points:
            for x, y in points:
                ax.plot(x, y, "o", color="red", markersize=8)

    # Remove axis ticks
    plt.axis("off")

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Segmentation visualization saved to {output_path}")

    plt.show()


def test_object_segmenter(
    image_path, box_coords=None, point_coords=None, output_path=None
):
    """Test the ObjectSegmenter with a sample image."""
    # Default box coordinates if none provided (x1, y1, x2, y2)
    if box_coords is None:
        box_coords = [100, 100, 400, 400]  # Default box in the middle of the image

    # Create and setup config manager
    config_manager = ConfigManager()
    config_manager.set_param(
        "model_settings.segmentation_model", "facebook/sam-vit-base"
    )

    # Initialize the ObjectSegmenter
    segmenter = ObjectSegmenter(config_manager)

    # Check if image path exists
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist")
        return

    # Load the image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)

    # Process point coordinates if provided
    input_points = None
    input_labels = None
    if point_coords:
        # Convert from flat list to list of coordinates
        points = []
        for i in range(0, len(point_coords), 2):
            if i + 1 < len(point_coords):
                points.append([point_coords[i], point_coords[i + 1]])

        input_points = [points]
        input_labels = [[1] * len(points)]  # All points are foreground

    # Process box coordinates
    input_boxes = [[box_coords[0], box_coords[1], box_coords[2], box_coords[3]]]

    # prepare for batch 
    images_batch = [image] * 3
    input_boxes_batch = [input_boxes] * 3

    # Segment objects in the image
    print("Segmenting object...")
    try:
        mask = segmenter.segment_object(
            images_data=images_batch,
            batch_input_boxes=input_boxes_batch,
            batch_input_points=input_points,
            batch_input_labels=input_labels,
        )

        # Print results
        print("\nSegmentation completed:")
        print(f"  Mask shape: {mask[0][0].shape}")
        print(f"  Mask unique values: {np.unique(mask[0][0])}")
        print(f"  Mask sum (pixels in mask): {mask[0][0].sum()}")

        return mask[0][0]

    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return None


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test ObjectSegmenter")
    parser.add_argument(
        "--image",
        default="recognize-anything/images/openset_example.jpg",
        type=str,
        help="Path to test image",
    )
    parser.add_argument(
        "--box",
        type=str,
        default="100,100,400,400",
        help="Comma-separated box coordinates (x1,y1,x2,y2)",
    )
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="Comma-separated point coordinates (x1,y1,x2,y2,...)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save visualization output"
    )

    # Parse arguments
    args = parser.parse_args()

    # Process box coordinates
    box_coords = [int(c.strip()) for c in args.box.split(",")]

    # Process point coordinates if provided
    point_coords = None
    if args.points:
        point_coords = [int(c.strip()) for c in args.points.split(",")]

    # Run the test
    test_object_segmenter(args.image, box_coords, point_coords, args.output)
