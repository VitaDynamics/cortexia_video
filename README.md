# Cortexia Video Project


## Overview
A Python-based video processing application for analyzing and processing video content.

## Features
- Video processing capabilities
- Configuration-based operation
- Extensible architecture

## Setup Instructions

### Prerequisites
- Python 3.10+
- Poetry (for dependency management)

### Installation
1. Clone this repository
2. Install dependencies:
```bash
poetry install
```

### Configuration
Place configuration files in the `config/` directory. Supported formats:
- config.yml
- config.json

### Running the Application
```bash
python main.py
```

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
