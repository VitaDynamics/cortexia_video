# Image Processing Scripts

This document describes the usage of various Python scripts located in the `scripts` folder. These scripts are designed to process image data, focusing on tasks such as depth estimation, image tagging, object detection, and segmentation. They can be used individually or as part of a larger workflow to prepare and analyze image datasets.

The scripts are generally designed to work with batches of images and often involve processing images within subfolders, treating each subfolder as a distinct dataset (e.g., frames from a video).

## Table of Contents

- [General Script Usage](#general-script-usage)
- [Script Details](#script-details)
  - [Stage 1: Image Tagging](#stage-1-image-tagging)
  - [Stage 2: Image Captioning](#stage-2-image-captioning)
  - [Stage 3: Object Detection & Segmentation](#stage-3-object-detection--segmentation)
  - [Stage 4: Depth Estimation](#stage-4-depth-estimation)
- [Batch Processing](#how-batch-processing-works)
- [Folder Structures](#target-folder-and-source-folder-structures-summary)

## General Script Usage

Many scripts in this folder share common conventions and command-line arguments.

### Common Command-Line Arguments

*   `--config <path_to_config.toml>`: Specifies the path to a configuration file (e.g., `config/heavy_mode.toml`). This file likely contains settings for models used in tasks like object detection, segmentation, or listing.
*   `--min-images <number>`: Used by scripts that process video folders (subfolders with sequences of images). This argument sets the minimum number of JPEG images a subfolder must contain to be processed. The default is typically 5.
*   `--folder <path_to_root_folder>`: Specifies the root directory that contains subfolders of images to be processed.
*   `--source-dir <path_to_source>`: Specifies the input directory for scripts like depth estimation. This directory can contain raw image files or ZIP archives of images.
*   `--target-dir <path_to_target>`: Specifies the output directory where results (like extracted images from ZIPs and generated depth maps) will be stored.

### Batch Processing

Several scripts leverage `batch_processor.py` to efficiently process large numbers of images. This utility loads images and processes them in batches, which can help manage memory and improve performance. While the `batch_size` is often hardcoded within individual scripts (e.g., to 4 or 8), the underlying `BatchProcessor` class is designed with batching as a core concept.

## Script Details

### Stage 1: Image Tagging

**Purpose**: Automatically identifies and tags passable areas or objects in images using Vision Language Models.

**Script**: `scripts/tag_images.py`

#### Command-Line Arguments

- `--folder` (required): Root folder containing video subfolders with images
- `--config` (optional): Path to configuration file (default: `config/heavy_mode.toml`)
- `--min-images` (optional): Minimum number of JPEG images required per subfolder (default: 5)
- `--categories` (optional): JSON string defining tag categories. This allows for customized, structured tagging instead of a simple keyword list.

#### Usage Examples

```bash
# Basic usage with default settings
python scripts/tag_images.py --folder /path/to/dataset

# Using custom configuration and categories
python scripts/tag_images.py \
    --folder /path/to/dataset \
    --config config/light_mode.toml \
    --categories '{"accessible_area": ["road", "sidewalk", "pathway"], "obstacles": ["vehicle", "cyclist", "pedestrian"]}'

# Process folders with fewer images
python scripts/tag_images.py \
    --folder /path/to/dataset \
    --min-images 3
```

#### Input Structure

```
dataset/
├── video_sequence_1/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── frame_003.jpg
└── video_sequence_2/
    ├── image_a.jpg
    └── image_b.jpg
```

#### Output

For each image, creates a JSON file with identified tags:

```
dataset/
├── video_sequence_1/
│   ├── frame_001.jpg
│   ├── frame_001_tag.json  # Generated tags
│   ├── frame_002.jpg
│   ├── frame_002_tag.json
│   └── ...
```

**Example tag file** (`frame_001_tag.json`):
```json
{
  "detectable_tags": {
    "accessible_area": ["road", "sidewalk"],
    "traffic_participants": ["vehicle"],
    "environmental_markers": ["lane marking"]
  },
  "tags": ["road", "sidewalk", "vehicle", "lane marking"],
  "category_map": {
    "road": "accessible_area",
    "sidewalk": "accessible_area",
    "vehicle": "traffic_participants",
    "lane marking": "environmental_markers"
  }
}
```

### Stage 2: Image Captioning

**Purpose**: To produce a short descriptive caption for every image frame.

**Script**: `scripts/caption_images.py`

#### Command-Line Arguments

* `--folder <path>`: (Required) Root directory containing subfolders of images (e.g., individual video clips). Each subfolder is processed independently.
* `--config <path>`: (Optional) Path to the configuration file specifying the captioning model. Defaults to `config/heavy_mode.toml`.
* `--min-images <number>`: (Optional) Minimum number of JPEG images a subfolder must contain to be processed. Defaults to `5`.

#### Input

* JPEG images stored in subfolders of the specified `--folder` directory.

#### Output

* For each image `frame.jpg`, a corresponding `frame_tag.json` file is created or updated to include a `"caption"` field containing the generated text.

#### Folder Structure Example

```
my_clips/
└── clip_01/
    ├── frame001.jpg
    ├── frame002.jpg
    └── ...
```

Running `python scripts/caption_images.py --folder my_clips` results in:

```
my_clips/
└── clip_01/
    ├── frame001.jpg
    ├── frame001_tag.json  # Includes { "caption": "..." }
    ├── frame002.jpg
    └── frame002_tag.json  # Includes { "caption": "..." }
```

### Stage 3: Object Detection & Segmentation

**Purpose**: Detects and segments objects in images based on tags from Stage 1.

**Script**: `scripts/detect_segment_images.py`

#### Prerequisites

- Images must have corresponding `_tag.json` files from Stage 1
- Tags will guide what objects to detect and segment
- A default list of common road-related tags is provided in `config/default_extra_tags.json`

#### Command-Line Arguments

- `--folder` (required): Root folder containing video subfolders with tagged images
- `--config` (optional): Path to configuration file (default: `config/heavy_mode.toml`)
- `--min-images` (optional): Minimum number of JPEG images required per subfolder (default: 5)
- `--extra-tags-file` (optional): JSON file containing tags that will be merged with those from Stage 1 before detection. Defaults to `config/default_extra_tags.json`.

#### Usage Examples

```bash
# Basic usage (requires tag files from Stage 1)
python scripts/detect_segment_images.py --folder /path/to/dataset

# Using custom configuration and tag file
python scripts/detect_segment_images.py \
    --folder /path/to/dataset \
    --config config/heavy_mode.toml \
    --extra-tags-file my_tags.json
```

#### Input Requirements

Each image must have a corresponding tag file:

```
dataset/
├── video_sequence_1/
│   ├── frame_001.jpg
│   ├── frame_001_tag.json  # Required from Stage 1
│   ├── frame_002.jpg
│   └── frame_002_tag.json  # Required from Stage 1
```

#### Output

Updates tag files with detection results and creates segmentation masks:

```
dataset/
├── video_sequence_1/
│   ├── frame_001.jpg
│   ├── frame_001_tag.json     # Updated with "objects" data
│   ├── frame_001_masks.npy    # Generated segmentation masks
│   ├── frame_002.jpg
│   ├── frame_002_tag.json     # Updated
│   └── frame_002_masks.npy    # Generated
```

**Updated tag file** (`frame_001_tag.json`):
```json
{
  "tags": ["road", "vehicle"],
  "category_map": {
    "road": "accessible_area",
    "vehicle": "traffic_participants"
  },
  "objects": [
    {
      "id": "uuid-1234-5678",
      "label": "vehicle",
      "score": 0.95,
      "box": [100, 150, 250, 300]
    }
  ]
}
```

**Mask file** (`frame_001_masks.npy`): Dictionary mapping object IDs to binary segmentation masks.

### Stage 4: Depth Estimation

**Purpose**: Generates depth maps for images using the DepthPro model.

**Script**: `scripts/batch_depth_estimation.py`

#### Command-Line Arguments

- `--folder` (required): Folder containing images to process
- `--recursive` (optional): Process subdirectories instead of the folder directly
- `--save_format` (optional): Format to save the depth map (`npy` or `png`). Default is `npy`.

#### Usage Examples

```bash
# Process images directly in a folder
python scripts/batch_depth_estimation.py --folder /path/to/images

# Process images in subdirectories (recommended for datasets from Stages 1-2)
python scripts/batch_depth_estimation.py --folder /path/to/dataset --recursive

# Save depth maps as PNG images
python scripts/batch_depth_estimation.py --folder /path/to/dataset --recursive --save_format png
```

#### Input

Supports JPG, JPEG, and PNG image formats:

```
dataset/
├── video_sequence_1/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
└── video_sequence_2/
    ├── image_a.jpg
    └── image_b.jpg
```

#### Output

Creates depth map files alongside original images. The output format depends on the `--save_format` argument.

**NPY Output (`--save_format npy`)**
```
dataset/
├── video_sequence_1/
│   ├── frame_001.jpg
│   ├── frame_001_depth_new.npy  # Generated depth map
│   ├── frame_002.jpg
│   ├── frame_002_depth_new.npy
│   └── ...
```
- **Depth files**: NumPy arrays (`.npy`) containing float32 depth values in meters, same resolution as input images.

**PNG Output (`--save_format png`)**
```
dataset/
├── video_sequence_1/
│   ├── frame_001.jpg
│   ├── frame_001_depth_new.png  # Generated depth map
│   ├── frame_002.jpg
│   ├── frame_002_depth_new.png
│   └── ...
```
- **Depth files**: PNG images (`.png`) where pixel values are normalized to 0-255.

### How Batch Processing Works (`batch_processor.py`)

The `batch_processor.py` script provides a generic framework for processing images in batches. This utility is used internally by other scripts like `batch_depth_estimation.py`, `tag_images.py`, and `detect_segment_images.py` to handle large datasets more efficiently.

**Purpose**: To provide a reusable class (`BatchProcessor`) that manages loading, batching, and applying inference and saving functions to images.

**Key Features**:

*   **Image Loading**: Loads images from a list of paths into an internal buffer.
*   **Batching**: Processes images in batches of a configurable size (e.g., `batch_size=4` or `batch_size=8`, often set within the calling script).
*   **Custom Functions**: Accepts custom `inference_func` (to perform the main processing task like depth estimation or tagging) and `save_func` (to store the results).
*   **Memory Management**: Includes mechanisms to clear the image buffer and invoke garbage collection, which is important when dealing with many high-resolution images.
*   **Image Collection Utilities**:
    *   `collect_images(folder)`: Collects all JPG/JPEG images from a given folder (non-recursive).
    *   `collect_images_recursive(root_dir)`: Collects all JPG/JPEG images recursively from a root directory and its subdirectories.

**Usage**:

Users typically don't run `batch_processor.py` directly. Instead, its `BatchProcessor` class and utility functions are imported and used by the other scripts. For example, when `detect_segment_images.py` processes a folder of images, it initializes a `BatchProcessor` instance, loads the images for that folder, and then calls `processor.process_batch()` with its specific detection/segmentation logic.

The use of batching helps in:

*   **Preventing Out-of-Memory Errors**: By loading and processing only a subset of images at a time.
*   **Improving Throughput**: Some models can achieve better performance when processing data in batches.

While the batch size is often pre-set in the main scripts, understanding that batch processing is happening can be useful for diagnosing performance or memory issues.

### Target Folder and Source Folder Structures (Summary)

This section summarizes the expected input (source) and output (target) folder structures for the main scripts.

**A. `batch_depth_estimation.py`**

*   **Source (`--source-dir`):**
    *   Can contain ZIP files with images:
        ```
        source_depth_zips/
        ├── batch_A.zip  # Contains image1.jpg, image2.jpg
        └── batch_B.zip  # Contains image3.jpg
        ```
    *   Or, can contain image files directly (though unzipping is a primary feature):
        ```
        source_depth_images/
        ├── image_X.jpg
        └── image_Y.jpg
        ```

*   **Target (`--target-dir`):**
    *   If source had ZIPs, they are extracted into subfolders within target:
        ```
        output_depth_from_zips/
        ├── batch_A/
        │   ├── image1.jpg        # Extracted
        │   ├── image1_depth.npy  # Output
        │   ├── image2.jpg        # Extracted
        │   └── image2_depth.npy  # Output
        └── batch_B/
            ├── image3.jpg        # Extracted
            └── image3_depth.npy  # Output
        ```
    *   If source had images directly, results are stored in target (structure within target depends on how `collect_images_recursive` finds them after the initial optional unzip step; typically flat if no zips were involved and source was flat):
        ```
        output_depth_from_images/ # Assuming source_dir had no zips.
        ├── image_X_depth.npy     # Output
        └── image_Y_depth.npy     # Output
        ```
        *(Note: The script first attempts to unzip everything from `--source-dir` to `--target-dir`, then processes images found recursively in `--target-dir`.)*

**B. `tag_images.py`**

*   **Source (`--folder`):** A root directory containing subfolders (e.g., "video" folders) of images.
    ```
    dataset_for_tagging/
    ├── video_01/
    │   ├── frame001.jpg
    │   ├── frame002.jpg
    │   └── ...
    └── video_02/
        ├── img_abc.jpg
        └── ...
    ```

*   **Output (within the same `--folder` structure):** `_tag.json` files are created alongside their respective images.
    ```
    dataset_for_tagging/
    ├── video_01/
    │   ├── frame001.jpg
    │   ├── frame001_tag.json  # Generated
    │   ├── frame002.jpg
    │   ├── frame002_tag.json  # Generated
    │   └── ...
    └── video_02/
        ├── img_abc.jpg
        ├── img_abc_tag.json   # Generated
        └── ...
    ```

**C. `detect_segment_images.py`**

*   **Source (`--folder`):** Similar to `tag_images.py`, but **requires** `_tag.json` files to be present, specifically with the `detectable_tags` field.
    ```
    dataset_for_detection/
    ├── scene_A/
    │   ├── image01.jpg
    │   ├── image01_tag.json  # Must exist, from tag_images.py
    │   │                     # Example: {"detectable_tags": ["car", "tree"], ...}
    │   ├── image02.jpg
    │   └── image02_tag.json  # Must exist
    └── scene_B/
        ├── photo_x.jpg
        └── photo_x_tag.json  # Must exist
    ```

*   **Output (within the same `--folder` structure):**
    *   `_tag.json` files are updated with an "objects" list.
    *   `_masks.npy` files are created for images where objects were detected.
    ```
    dataset_for_detection/
    ├── scene_A/
    │   ├── image01.jpg
    │   ├── image01_tag.json  # Updated with "objects"
    │   ├── image01_masks.npy # Generated segmentation masks
    │   ├── image02.jpg
    │   └── image02_tag.json  # Updated
    │   └── image02_masks.npy # Generated
    └── scene_B/
        ├── photo_x.jpg
        ├── photo_x_tag.json  # Updated
        └── photo_x_masks.npy # Generated
    ```

This summary should help in preparing your data correctly for each script and understanding where to find the results.
