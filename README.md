# Cortexia Video

Cortexia Video is a deep learning toolkit for analyzing video content. It combines object listing, detection, segmentation, depth estimation, and optional description generation into a comprehensive command line workflow.

## Features
- **Object listing** – identify objects present in frames using a vision language model.
- **Detection** – locate objects with Grounding DINO.
- **Segmentation** – segment objects using the SAM model.
- **Depth estimation** – generate depth maps from images using the DepthPro model.
- **Description** – (optional) describe segments with NVIDIA DAM.
- **Feature extraction** – extract CLIP features for scenes or objects.
- **Batch processing** – efficient batch processing with memory management and error handling.

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for depth estimation)
- Git LFS for downloading model checkpoints

### Setup
1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/DylanLIiii/cortexia_video.git
cd cortexia_video
```

2. Install the dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r uv.lock
```

3. Download the DepthPro model checkpoint:
```bash
# The checkpoint should be placed at checkpoints/depth_pro.pt
# This is typically handled automatically by the depth_pro library
```

## Configuration
Configuration files live in the `config/` directory. Example TOML files are provided:

- `example_config.toml` – balanced defaults
- `light_mode.toml` – faster models
- `heavy_mode.toml` – higher quality models

Each configuration defines model names and processing options. You can point the application at any custom file via `--config`.

## Usage

### Main Video Processing Pipeline
Run the CLI with the desired configuration:

```bash
python main.py --config config/example_config.toml
```

The configuration option `processing.default_mode` controls the pipeline. Modes are separated with `|` and may include:

- `list` – list objects in frames
- `detect` – detect objects
- `segment` – segment detected boxes
- `describe` – generate descriptions for segments
- `extract_scene` – extract CLIP features for the full frame
- `extract_object` – extract features for objects

### Batch Depth Estimation

The project includes a powerful batch depth estimation tool for processing multiple images efficiently:

#### Basic Usage

Process all images in a folder:
```bash
python scripts/batch_depth_estimation.py --folder /path/to/images
```

Process images in subdirectories recursively:
```bash
python scripts/batch_depth_estimation.py --folder /path/to/parent_folder --recursive
```

#### Features
- **Efficient batch processing**: Processes images in configurable batches (default: 4 images)
- **Memory management**: Automatic cleanup and garbage collection
- **Error handling**: Graceful failure recovery for individual images
- **GPU acceleration**: Automatic CUDA detection and usage
- **Flexible input**: Supports JPG, JPEG, and PNG formats

#### Output
- Depth maps saved as NumPy arrays (`.npy` files)
- Files named as `<original_name>_depth_new.npy`
- Float32 depth values in meters
- Same resolution as input images

#### Example Output
```bash
Found 15 images for processing in photos
Loading images into buffer...
Loaded 15 images
Starting batch processing for folder photos...
Processing batch 1/4
Infering 4 images
Infered 4 images
Saved ./photos/image1_depth_new.npy
Saved ./photos/image2_depth_new.npy
...
```

For detailed documentation, see [docs/batch_depth_estimation_usage.md](docs/batch_depth_estimation_usage.md).

## Output
Results are written to the directory specified by `processing.output_directory` in the configuration. When visualization is enabled, annotated frames are saved alongside JSON annotations.

## Project Structure
- `cortexia_video/` - Main source package
  - `depth_estimation.py` - DepthPro model wrapper for batch depth estimation
  - `object_detection.py` - Grounding DINO integration
  - `object_segmentation.py` - SAM model integration
  - `feature_extraction.py` - CLIP feature extraction
  - `object_listing.py` - Vision language model for object identification
- `config/` - Configuration files
- `docs/` - Project documentation
  - `batch_depth_estimation_usage.md` - Detailed depth estimation documentation
- `scripts/` - Project scripts and utilities
  - `batch_depth_estimation.py` - Batch depth estimation script
  - `batch_processor.py` - Generic batch processing utilities
  - `examples/` - Usage examples and demonstrations
- `checkpoints/` - Model checkpoints and weights
- `ml-depth-pro/` - DepthPro model implementation (submodule)

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computations
- **OpenCV**: Computer vision operations
- **Transformers**: Hugging Face model library

### Depth Estimation Dependencies
- **depth_pro**: DepthPro model library
- **timm**: PyTorch image models
- **einops**: Tensor operations

### Optional Dependencies
- **CUDA**: GPU acceleration (recommended)
- **Matplotlib**: Visualization and plotting
- **Wandb**: Experiment tracking

## Performance Considerations

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support recommended for optimal performance
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large batches
- **Storage**: SSD recommended for faster I/O operations

### Optimization Tips
- Adjust batch size based on available GPU memory
- Use GPU acceleration when available
- Process images in appropriate resolutions for your use case
- Monitor memory usage during batch processing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in scripts
   - Close other GPU-intensive applications
   - Use CPU processing as fallback

2. **Model Loading Errors**
   - Verify all dependencies are installed
   - Check model checkpoint availability
   - Ensure proper CUDA installation

3. **Image Processing Errors**
   - Verify image file formats are supported
   - Check file permissions and paths
   - Ensure sufficient disk space for output

For more detailed troubleshooting, refer to the documentation in the `docs/` directory.
