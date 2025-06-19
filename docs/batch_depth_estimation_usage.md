# Batch Depth Estimation Script Documentation

## Overview

The `batch_depth_estimation.py` script provides a command-line interface for performing depth estimation on batches of images using the DepthPro model. It efficiently processes images in batches to estimate depth maps and save them as NumPy arrays.

## Location

```
scripts/batch_depth_estimation.py
```

## Features

- **Batch Processing**: Processes multiple images efficiently in configurable batch sizes
- **Memory Management**: Automatic memory cleanup and garbage collection
- **Error Handling**: Robust error handling with graceful failure recovery
- **Flexible Input**: Can process images directly in a folder or recursively through subdirectories
- **Model Reuse**: Global model instance to avoid reloading the depth estimation model
- **Progress Tracking**: Detailed console output showing processing progress

## Dependencies

- `numpy`: For saving depth maps as .npy files
- `PIL (Pillow)`: For image loading and processing
- `torch`: Required by the depth estimation model
- `depth_pro`: The depth estimation model library
- `cortexia_video.depth_estimation`: Custom depth estimator wrapper
- `batch_processor`: Custom batch processing utilities

## Usage

### Basic Syntax

```bash
python scripts/batch_depth_estimation.py --folder <path_to_folder> [--recursive]
```

### Arguments

- `--folder` (required): Path to the folder containing images to process
- `--recursive` (optional): Process subdirectories instead of the folder directly

### Processing Modes

#### 1. Direct Folder Processing (Default)

Processes all images directly in the specified folder:

```bash
python scripts/batch_depth_estimation.py --folder /path/to/images
```

This will:
- Find all images (JPG, JPEG, PNG) in the specified folder
- Process them in batches of 4 images
- Save depth maps as `<original_name>_depth_new.npy` files

#### 2. Recursive Subdirectory Processing

Processes images in each subdirectory separately:

```bash
python scripts/batch_depth_estimation.py --folder /path/to/parent_folder --recursive
```

This will:
- Iterate through each subdirectory in the parent folder
- Process images in each subdirectory as a separate batch
- Skip subdirectories that contain no images

## Output

For each successfully processed image, the script generates:
- A depth map saved as `<original_filename>_depth_new.npy`
- Console output showing processing progress

### Output Format

The depth maps are saved as NumPy arrays (.npy files) containing:
- **Shape**: Same height and width as the input image
- **Data Type**: Float32
- **Values**: Depth values in meters (or model-specific units)

## Examples

### Example 1: Process Images in a Single Folder

```bash
# Process all images in the 'photos' directory
python scripts/batch_depth_estimation.py --folder ./photos

# Expected output:
# Found 15 images for processing in photos
# Loading images into buffer...
# Loaded 15 images
# Starting batch processing for folder photos...
# Processing batch 1/4
# Infering 4 images
# Infered 4 images
# Saved ./photos/image1_depth_new.npy
# Saved ./photos/image2_depth_new.npy
# ...
```

### Example 2: Process Multiple Video Folders

```bash
# Process images in each subdirectory of 'video_dataset'
python scripts/batch_depth_estimation.py --folder ./video_dataset --recursive

# Expected output:
# Found 20 images for processing in video_001
# Processing subdirectory video_001...
# Found 18 images for processing in video_002
# Processing subdirectory video_002...
# ...
```

## Configuration

### Batch Size

The script uses a default batch size of 4 images. This is configured in the `process_folder_images()` function:

```python
processor = BatchProcessor(batch_size=4)
```

To modify the batch size, edit line 118 in the script. Consider your GPU memory when adjusting this value.

### Supported Image Formats

The script automatically detects and processes:
- `.jpg`, `.jpeg`, `.JPG`, `.JPEG`
- `.png`, `.PNG`

## Error Handling

The script includes comprehensive error handling:

1. **Image Loading Errors**: Skips corrupted or unreadable images
2. **Inference Errors**: Returns empty results for failed estimations
3. **Save Errors**: Continues processing other images if saving fails
4. **Memory Management**: Always cleans up resources, even on errors

## Performance Considerations

- **GPU Usage**: Automatically uses CUDA if available, falls back to CPU
- **Memory Management**: Clears image buffers after processing to prevent memory leaks
- **Batch Processing**: Processes images in batches to optimize GPU utilization
- **Model Reuse**: Loads the depth estimation model once and reuses it

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size in the script
   - Ensure sufficient GPU/system memory

2. **No Images Found**
   - Check folder path is correct
   - Verify images are in supported formats
   - Ensure proper file permissions

3. **Model Loading Errors**
   - Verify depth_pro library is properly installed
   - Check CUDA installation if using GPU

### Debug Output

The script provides detailed console output including:
- Number of images found
- Processing progress for each batch
- Success/failure status for each image
- File paths for saved depth maps

## Integration

This script is part of the cortexia_video processing pipeline and can be integrated with other batch processing workflows. It follows the same patterns as other batch processing scripts in the `scripts/` directory.

## Related Files

- `scripts/batch_processor.py`: Core batch processing utilities
- `cortexia_video/depth_estimation.py`: Depth estimation model wrapper
- `scripts/examples/batch_depth_example.py`: Usage example
