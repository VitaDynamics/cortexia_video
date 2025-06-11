# Cortexia Video

Cortexia Video is a deep learning toolkit for analyzing video content. It combines object listing, detection, segmentation and optional description generation into a single command line workflow.

## Features
- **Object listing** – identify objects present in frames using a vision language model.
- **Detection** – locate objects with Grounding DINO.
- **Segmentation** – segment objects using the SAM model.
- **Description** – (optional) describe segments with NVIDIA DAM.
- **Feature extraction** – extract CLIP features for scenes or objects.

## Installation
1. Clone the repository
2. Install the dependencies using [uv](https://github.com/astral-sh/uv)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r uv.lock
```

## Configuration
Configuration files live in the `config/` directory. Example TOML files are provided:

- `example_config.toml` – balanced defaults
- `light_mode.toml` – faster models
- `heavy_mode.toml` – higher quality models

Each configuration defines model names and processing options. You can point the application at any custom file via `--config`.

## Running
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

## Output
Results are written to the directory specified by `processing.output_directory` in the configuration. When visualization is enabled, annotated frames are saved alongside JSON annotations.

## Project Structure
- `cortexia_video/` - Main source package
- `config/` - Configuration files
- `docs/` - Project documentation
- `scripts/` - Project scripts and utilities

### Batch Depth Estimation Script
Use `scripts/batch_depth_estimation.py` to extract images from zip archives and compute depth maps for each JPG image.

Example:
```bash
python scripts/batch_depth_estimation.py --source-dir /path/to/zip_folder --target-dir /path/to/workdir
```
