# Installation Guide

This guide provides detailed instructions for installing the Cortexia Video SDK on various platforms and environments.

## System Requirements

### Minimum Requirements
- **Python**: 3.10 or higher
- **RAM**: 16GB (8GB minimum for basic usage)
- **Storage**: 30GB free disk space for models and processing
- **GPU**: CUDA-compatible GPU recommended (CPU fallback available)

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 32GB or more
- **Storage**: 100GB+ SSD for optimal performance
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.8 or higher

### Operating Systems
- **Linux**: Ubuntu 20.04+, CentOS 8+, Rocky Linux 8+
- **macOS**: macOS 12+ (Monterey or later)
- **Windows**: Windows 10/11 (WSL2 recommended)

## Installation Methods

### Method 1: Using UV (Recommended)

UV is a fast Python package manager that handles dependencies efficiently.

```bash
# Clone the repository
git clone --recursive https://github.com/DylanLIiii/cortexia.git
cd cortexia

# Install using UV
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### Method 2: Using PIP

```bash
# Clone the repository
git clone --recursive https://github.com/DylanLIiii/cortexia.git
cd cortexia

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

### Method 3: Using Conda

```bash
# Create conda environment
conda create -n cortexia python=3.10
conda activate cortexia

# Clone repository
git clone --recursive https://github.com/DylanLIiii/cortexia.git
cd cortexia

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl

# Install CUDA (if using GPU)
# Follow NVIDIA's official guide for your Ubuntu version
# https://developer.nvidia.com/cuda-downloads

# Clone and install
git clone --recursive https://github.com/DylanLIiii/cortexia.git
cd cortexia
uv sync
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10+
brew install python@3.10

# Clone and install
git clone --recursive https://github.com/DylanLIiii/cortexia.git
cd cortexia
uv sync
```

### Windows (WSL2)

```bash
# Install WSL2
wsl --install

# Install Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2 Ubuntu:
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-dev python3.10-venv python3-pip build-essential cmake git

# Clone and install
git clone --recursive https://github.com/DylanLIiii/cortexia.git
cd cortexia
uv sync
```

### Windows (Native)

```powershell
# Install Python 3.10+ from https://python.org
# Make sure to check "Add Python to PATH" during installation

# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select "Desktop development with C++"

# Clone repository (using Git for Windows)
git clone --recursive https://github.com/DylanLIiii/cortexia.git
cd cortexia

# Install in development mode
pip install -e .
```

## Environment Setup

### Environment Variables

For users in China or regions with restricted access to Hugging Face models:

```bash
# Set Hugging Face mirror
export HF_HOME=/path/to/model/cache
export HF_ENDPOINT=https://hf-mirror.com

# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export HF_HOME=/path/to/model/cache' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

### GPU Configuration

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Model Cache Configuration

```bash
# Set custom model cache directory
export HF_HOME=/vita-vepfs-data/fileset1/usr_data/min.dong/model/huggingface
export TRANSFORMERS_CACHE=/vita-vepfs-data/fileset1/usr_data/min.dong/model/transformers
export TORCH_HOME=/vita-vepfs-data/fileset1/usr_data/min.dong/model/torch
```

## Verification

### Verify Installation

```bash
# Test Python version
python --version
# Should show Python 3.10.x

# Test package installation
python -c "import cortexia; print('Cortexia SDK installed successfully')"

# Test CLI
cortexia-video --help

# Test core components
python -c "
import cortexia
print('Available features:', cortexia.list_features())
print('Available gates:', cortexia.list_gates())
"
```

### Test GPU Support

```bash
# Test PyTorch CUDA support
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    print('Current CUDA device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name())
"
```

### Test Model Loading

```bash
# Test basic feature creation
python -c "
import cortexia
try:
    detector = cortexia.create_feature('detection')
    print('✓ Detection feature created successfully')
except Exception as e:
    print('✗ Failed to create detection feature:', e)

try:
    blur_gate = cortexia.create_gate('blur')
    print('✓ Blur gate created successfully')
except Exception as e:
    print('✗ Failed to create blur gate:', e)
"
```

## Configuration Files

### Default Configuration

The SDK includes several configuration files in the `config/` directory:

- **`config/example_config.toml`**: Balanced configuration for general use
- **`config/light_mode.toml`**: Lightweight configuration for faster processing
- **`config/heavy_mode.toml`**: High-quality configuration with larger models

### Test Configuration

```bash
# Test configuration loading
python -c "
from cortexia.core.config.manager import ConfigManager
config = ConfigManager('config/example_config.toml')
config.load_config()
print('✓ Configuration loaded successfully')
print('Batch size:', config.get_param('processing.batch_size', 4))
"
```

## Troubleshooting

### Common Installation Issues

#### Python Version Issues

```bash
# Check Python version
python --version

# If not 3.10+, install correct version
# Ubuntu/Debian
sudo apt install python3.10 python3.10-dev python3.10-venv

# macOS
brew install python@3.10

# Create virtual environment with correct Python version
python3.10 -m venv cortexia_env
source cortexia_env/bin/activate
```

#### Dependency Conflicts

```bash
# Clean install in fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install -e .

# If conflicts persist, try
pip install -e . --no-deps
pip install -r requirements.txt
```

#### CUDA Issues

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# If CUDA not found, install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or reinstall with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Model Download Issues

```bash
# Check internet connection and Hugging Face access
curl -I https://huggingface.co

# Set custom cache if default location has permission issues
export HF_HOME=$HOME/.cache/huggingface
mkdir -p $HF_HOME
chmod 755 $HOME/.cache/huggingface

# For Chinese users, use mirror
export HF_ENDPOINT=https://hf-mirror.com
```

#### Memory Issues

```bash
# Reduce batch size in configuration
cat > config/low_memory_config.toml << EOF
[processing]
batch_size = 1
frame_interval = 60

[model_settings]
# Use smaller models if needed
object_detection_model = "YOLO-World/small"
EOF

# Test with low memory config
python -c "
from cortexia.core.config.manager import ConfigManager
config = ConfigManager('config/low_memory_config.toml')
config.load_config()
print('✓ Low memory configuration loaded')
"
```

### Performance Optimization

#### GPU Memory Management

```bash
# Set environment variables for memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Use mixed precision if supported
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"  # Adjust for your GPU
```

#### I/O Optimization

```bash
# Use SSD for model cache
export HF_HOME=/path/to/ssd/huggingface_cache

# Increase file descriptor limit
ulimit -n 65536
```

#### Parallel Processing

```bash
# Set number of workers for data loading
export NUM_WORKERS=4

# Enable multiprocessing
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Docker Installation

### Using Pre-built Docker Image

```bash
# Pull the image
docker pull ghcr.io/vitadynamics/cortexia:latest

# Run the container
docker run --gpus all -it \
    -v /path/to/models:/models \
    -v /path/to/data:/data \
    -v /path/to/output:/output \
    ghcr.io/vitadynamics/cortexia:latest
```

### Building from Source

```bash
# Clone repository
git clone --recursive https://github.com/DylanLIiii/cortexia.git
cd cortexia

# Build Docker image
docker build -t cortexia:latest .

# Run the container
docker run --gpus all -it \
    -v $(pwd)/config:/app/config \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    cortexia:latest
```

## Next Steps

After successful installation:

1. **Read Getting Started**: Follow the [Getting Started](getting_started.md) guide
2. **Explore Examples**: Check the [cookbook/](../../cookbook/) directory
3. **Configure Your Setup**: See the [Configuration](configuration.md) guide
4. **Test Your Setup**: Run the verification tests above

## Need Help?

If you encounter any issues during installation:

1. **Check the logs**: Look for error messages in the console output
2. **Review requirements**: Ensure all system requirements are met
3. **Try fresh environment**: Create a new virtual environment
4. **Check documentation**: Review this guide and troubleshooting section
5. **Report issues**: Create an issue on GitHub with detailed error information

For additional support, please refer to the [Troubleshooting](troubleshooting.md) guide or create an issue on the repository.