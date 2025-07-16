# Image Processing Scripts

This document describes the usage of various Python scripts located in the `scripts` folder. These scripts are designed to process image data, focusing on tasks such as depth estimation, image tagging, object detection, and segmentation. They can be used individually or as part of a larger workflow to prepare and analyze image datasets.

The scripts are generally designed to work with batches of images and often involve processing images within subfolders, treating each subfolder as a distinct dataset (e.g., frames from a video).
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

### 1. Depth Estimation (`batch_depth_estimation.py`)

This script performs batch depth estimation for images. It can process images directly or extract them from ZIP archives.

**Purpose:** To generate depth maps for a collection of images.

**Command-Line Arguments:**

*   `--source-dir <path>`: (Required) Directory containing image files (e.g., `.jpg`, `.jpeg`) or ZIP archives. If ZIP files are present, they will be extracted.
*   `--target-dir <path>`: (Required) Directory where extracted images (if any) and the output depth maps will be stored. If images are extracted from a ZIP, a subfolder named after the ZIP file (without extension) will be created in the `target-dir` to store these images and their corresponding depth maps.
*   `--force`: (Optional) If specified, existing extracted folders in the `target-dir` will be overwritten.

**Input:**

*   Image files (JPG, JPEG) located directly in `source-dir` or within ZIP archives in `source-dir`.

**Output:**

*   Depth maps are saved as NumPy `.npy` files. For an input image `imagename.jpg`, the corresponding depth map will be `imagename_depth.npy`.
*   These output files are saved in the `target-dir`. If images were extracted from a ZIP file (e.g., `batch1.zip`), they will be placed inside a corresponding subfolder (`target-dir/batch1/imagename_depth.npy`). If images were processed directly from `source-dir` (not from a ZIP), their depth maps will be in `target-dir` (though the script seems to primarily expect ZIPs or a flat structure that it then processes recursively *within* target-dir after unzipping). The script uses `collect_images_recursive(args.target_dir)`, so outputs will mirror the structure found/created in `target-dir`.

**Folder Structure Example:**

If `source-dir` contains `dataset.zip`, and `dataset.zip` contains `image1.jpg` and `image2.jpg`:

```
source_dir/
└── dataset.zip

# Command: python scripts/batch_depth_estimation.py --source-dir source_dir --target-dir output_depth
# (Assuming DepthProEstimator is available and configured)

output_depth/
└── dataset/
    ├── image1.jpg        # Extracted image
    ├── image1_depth.npy  # Generated depth map
    ├── image2.jpg        # Extracted image
    └── image2_depth.npy  # Generated depth map
```

If `source-dir` contains images directly (less common usage based on script's unzipping focus):

```
source_dir/
├── imageA.jpg
└── imageB.jpg

# Command: python scripts/batch_depth_estimation.py --source-dir source_dir --target-dir output_depth

output_depth/ # The script unzips to target_dir, then collects images recursively from target_dir.
              # If source-dir had no zips, it would effectively be like target-dir starts with those images.
├── imageA_depth.npy
└── imageB_depth.npy
```
*(Note: The script first unzips everything from `source-dir` to `target-dir` and then recursively searches `target-dir` for images. The primary use case seems to be processing ZIP archives.)*
### 2. Image Tagging (`tag_images.py`)

This script is used to automatically generate initial descriptive tags for images, with a specific focus on identifying passable areas or surfaces within the images. These tags are saved in JSON files and can serve as input for subsequent detection and segmentation tasks.

**Purpose:** To automatically annotate images with a list of textual tags, primarily identifying drivable or passable areas.

**Command-Line Arguments:**

*   `--folder <path>`: (Required) The root folder containing subfolders (e.g., video folders). Each subfolder should contain the images to be tagged.
*   `--config <path>`: (Optional) Path to a configuration file (e.g., `config/heavy_mode.toml`) that specifies the Vision Language Model (VLM) or other models used for object listing. Defaults to `config/heavy_mode.toml`.
*   `--min-images <number>`: (Optional) Minimum number of JPEG images a subfolder must contain to be processed. Defaults to 5.
*   `--categories <json_string>`: (Optional) A JSON string that defines categories of tags, separating them into "detectable" and "undetectable" groups. This allows the subsequent detection script to focus only on relevant objects.
    *   Example: `'{"detectable_tags": ["car", "person", "road"], "undetectable_tags": ["daylight", "cloudy", "building"]}'`

**Input:**

*   Image files (JPG, JPEG) located within subfolders of the specified `--folder`. For example:
    ```
    processing_root/
    └── video1_frames/
        ├── frame001.jpg
        ├── frame002.jpg
        └── ...
    └── video2_frames/
        ├── image_abc.jpg
        └── ...
    ```

**Output:**

*   For each processed image (e.g., `imagename.jpg`), a JSON file named `imagename_tag.json` is created in the same subfolder.
*   This JSON file contains categorized tags and a list of tags that are considered detectable.
*   Example `imagename_tag.json`:
    ```json
    {
      "tags": {
        "detectable_tags": ["road", "car"],
        "undetectable_tags": ["building", "sky"]
       },
      "category_map": {
        "road": "accessible_area",
        "car": "traffic_participants",
        "building": "other",
        "sky": "other"
      },
      "detectable_tags": ["road", "car"]
    }
    ```

**Folder Structure Example:**

Given the input structure:
```
my_dataset/
└── sequence1/
    ├── img_001.jpg
    └── img_002.jpg
```

After running `python scripts/tag_images.py --folder my_dataset`:
```
my_dataset/
└── sequence1/
    ├── img_001.jpg
    ├── img_001_tag.json  # Generated tags
    ├── img_002.jpg
    └── img_002_tag.json  # Generated tags
```
*(This script processes images in subfolders of the main `--folder`.)*
### 3. Auto Tagging with Object Detection and Segmentation (`detect_segment_images.py`)

This script performs object detection and segmentation on images based on pre-existing tags. It updates the tag files with information about detected objects (bounding boxes, scores, labels) and saves segmentation masks.

**Purpose:** To identify and segment objects in images using tags provided in `_tag.json` files.

**Command-Line Arguments:**

*   `--folder <path>`: (Required) The root folder containing subfolders (e.g., video folders). Each subfolder should have images and their corresponding `_tag.json` files.
*   `--config <path>`: (Optional) Path to a configuration file (e.g., `config/heavy_mode.toml`) specifying the models for detection and segmentation. Defaults to `config/heavy_mode.toml`.
*   `--min-images <number>`: (Optional) Minimum number of JPEG images a subfolder must contain to be processed. Defaults to 5.

**Input:**

*   Image files (JPG, JPEG) located within subfolders of the specified `--folder`.
*   A corresponding `imagename_tag.json` file for each image. This file must contain a `detectable_tags` field, which is a list of strings. The script uses this list to determine which objects to detect and segment.
    *   Example `imagename_tag.json` (input):
        ```json
        {
          "tags": {
            "detectable_tags": ["road", "car"],
            "undetectable_tags": ["building", "sky"]
          },
          "category_map": {
            "road": "accessible_area",
            "car": "traffic_participants",
            "building": "other",
            "sky": "other"
          },
          "detectable_tags": ["road", "car"]
        }
        ```

**Output:**

1.  **Updated `imagename_tag.json` files:** The script adds an "objects" key to the JSON file. The value is a list of detected objects, where each object has:
    *   `id`: A unique UUID for the object.
    *   `label`: The class label of the detected object (e.g., "car", "road").
    *   `score`: The confidence score of the detection.
    *   `box`: The bounding box coordinates `[x_min, y_min, x_max, y_max]`.
    *   Example updated `imagename_tag.json`:
        ```json
        {
          "tags": {
            "detectable_tags": ["road", "car"],
            "undetectable_tags": ["building", "sky"]
          },
          "category_map": {
            "road": "accessible_area",
            "car": "traffic_participants",
            "building": "other",
            "sky": "other"
          },
          "detectable_tags": ["car"],
          "objects": [
            {
              "id": "some-uuid-1234",
              "label": "car",
              "score": 0.95,
              "box": [100, 150, 250, 300]
            }
          ]
        }
        ```
2.  **Segmentation Masks (`imagename_masks.npy`):** For each image where objects are detected, a `.npy` file is created (e.g., `imagename_masks.npy`). This file stores a Python dictionary where:
    *   Keys are the object `id`s (from the "objects" list in the JSON file).
    *   Values are the corresponding binary segmentation masks (NumPy arrays).

**Folder Structure Example:**

Given the input structure:
```
my_dataset/
└── sequence1/
    ├── img_001.jpg
    ├── img_001_tag.json  # Containing {"tags": ["car"], "category_map": {"car": "traffic_participants"}}
    └── img_002.jpg
        └── img_002_tag.json  # Containing {"tags": ["road"], "category_map": {"road": "accessible_area"}}
```

After running `python scripts/detect_segment_images.py --folder my_dataset`:
```
my_dataset/
└── sequence1/
    ├── img_001.jpg
    ├── img_001_tag.json  # Updated with "objects"
    ├── img_001_masks.npy # Generated masks for objects in img_001
    └── img_002.jpg
        ├── img_002_tag.json  # Updated with "objects"
        └── img_002_masks.npy # Generated masks for objects in img_002
```
*(Note: If an `imagename_tag.json` file is missing or its tag list is empty for a given image, that image might be skipped, and no `_masks.npy` file will be generated for it.)*

### 4. Image Captioning (`caption_images.py`)

This script generates natural-language captions for each image and stores them in the existing
`_tag.json` files. If a tag file does not yet exist for an image, it will be created automatically.

**Purpose:** To produce a short descriptive caption for every image frame.

**Command-Line Arguments:**

* `--folder <path>`: (Required) Root directory containing subfolders of images (e.g., individual
  video clips). Each subfolder is processed independently.
* `--config <path>`: (Optional) Path to the configuration file specifying the captioning model.
  Defaults to `config/heavy_mode.toml`.
* `--min-images <number>`: (Optional) Minimum number of JPEG images a subfolder must contain to be
  processed. Defaults to `5`.

**Input:**

* JPEG images stored in subfolders of the specified `--folder` directory.

**Output:**

* For each image `frame.jpg`, a corresponding `frame_tag.json` file is created or updated to include
  a `"caption"` field containing the generated text.

**Folder Structure Example:**

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

### 5. How Batch Processing Works (`batch_processor.py`)

The `batch_processor.py` script provides a generic framework for processing images in batches. This utility is used internally by other scripts like `batch_depth_estimation.py`, `tag_images.py`, and `detect_segment_images.py` to handle large datasets more efficiently.

**Purpose:** To provide a reusable class (`BatchProcessor`) that manages loading, batching, and applying inference and saving functions to images.

**Key Features:**

*   **Image Loading:** Loads images from a list of paths into an internal buffer.
*   **Batching:** Processes images in batches of a configurable size (e.g., `batch_size=4` or `batch_size=8`, often set within the calling script).
*   **Custom Functions:** Accepts custom `inference_func` (to perform the main processing task like depth estimation or tagging) and `save_func` (to store the results).
*   **Memory Management:** Includes mechanisms to clear the image buffer and invoke garbage collection, which is important when dealing with many high-resolution images.
*   **Image Collection Utilities:**
    *   `collect_images(folder)`: Collects all JPG/JPEG images from a given folder (non-recursive).
    *   `collect_images_recursive(root_dir)`: Collects all JPG/JPEG images recursively from a root directory and its subdirectories.

**Usage:**

Users typically don't run `batch_processor.py` directly. Instead, its `BatchProcessor` class and utility functions are imported and used by the other scripts. For example, when `detect_segment_images.py` processes a folder of images, it initializes a `BatchProcessor` instance, loads the images for that folder, and then calls `processor.process_batch()` with its specific detection/segmentation logic.

The use of batching helps in:

*   **Preventing Out-of-Memory Errors:** By loading and processing only a subset of images at a time.
*   **Improving Throughput:** Some models can achieve better performance when processing data in batches.

While the batch size is often pre-set in the main scripts, understanding that batch processing is happening can be useful for diagnosing performance or memory issues.
### 6. Target Folder and Source Folder Structures (Summary)

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
