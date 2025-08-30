# Configuration Guide

This guide provides comprehensive information about configuring the Cortexia Video SDK for various use cases and environments.

## Configuration Overview

Cortexia Video SDK uses TOML configuration files for flexible and human-readable configuration. The configuration system supports:

- **Environment-specific settings** (development, production, etc.)
- **Component-specific parameters** for features, gates, and samplers
- **Model selection** and hyperparameter tuning
- **Performance optimization** settings
- **Logging and monitoring** configuration

## Configuration File Structure

### Basic Configuration File

```toml
# Basic configuration structure
[logging]
level = "INFO"
file = "cortexia.log"

[model_settings]
# Model selection
object_listing_model = "Qwen/Qwen2.5-VL-3B-Instruct"
object_detection_model = "IDEA-Research/grounding-dino-base"
segmentation_model = "facebook/sam-vit-huge"

[processing]
# Processing parameters
batch_size = 4
frame_interval = 30
input_video_path = "sample_data/"
output_directory = "output/"

[detection_settings]
# Detection-specific parameters
box_threshold = 0.3
text_threshold = 0.3
```

### Configuration Sections

| Section | Purpose | Key Parameters |
|---------|---------|----------------|
| `logging` | Logging configuration | `level`, `file`, `format` |
| `model_settings` | Model selection | Various model names and paths |
| `processing` | General processing | `batch_size`, `frame_interval`, paths |
| `detection_settings` | Object detection | `box_threshold`, `text_threshold` |
| `description_settings` | Scene description | `temperature`, `top_p`, `max_tokens` |
| `visualization` | Output visualization | `enabled`, `image_format`, colors |

## Available Configuration Files

The SDK includes several pre-configured files in the `config/` directory:

### 1. Example Configuration (`config/example_config.toml`)

Balanced configuration for general use:

```toml
[logging]
level = "INFO"
file = "app.log"

[model_settings]
object_listing_model = "Qwen/Qwen2.5-VL-3B-Instruct"
object_detection_model = "IDEA-Research/grounding-dino-base"
segmentation_model = "facebook/sam-vit-huge"
description_model = "nvidia/DAM-3B-Self-Contained"
clip_feature_model = "PE-Core-B16-224"
image_captioning_model = "vikhyatk/moondream2"

[detection_settings]
box_threshold = 0.3
text_threshold = 0.3

[description_settings]
temperature = 0.2
top_p = 0.5
num_beams = 1
max_tokens = 512

[processing]
default_mode = "list | detect | segment | extract_scene | extract_object"
input_video_path = "sample_data/"
output_directory = "output/"
frame_interval = 50
batch_size = 2
image_format = "jpg"

[visualization]
enabled = true
annotated_image_format = "jpg"
contour_enabled = true
contour_thickness = 3
description_viz_enabled = false
```

### 2. Light Mode (`config/light_mode.toml`)

Lightweight configuration for faster processing:

```toml
[logging]
level = "WARNING"

[model_settings]
object_listing_model = "YOLO-World/small"
object_detection_model = "YOLO-World/small"
segmentation_model = "facebook/sam-vit-base"
description_model = "vikhyatk/moondream2"

[processing]
batch_size = 8
frame_interval = 100
default_mode = "detect | segment"

[detection_settings]
box_threshold = 0.4
text_threshold = 0.4

[visualization]
enabled = false
```

### 3. Heavy Mode (`config/heavy_mode.toml`)

High-quality configuration with larger models:

```toml
[logging]
level = "DEBUG"

[model_settings]
object_listing_model = "Qwen/Qwen2.5-VL-7B-Instruct"
object_detection_model = "IDEA-Research/grounding-dino-large"
segmentation_model = "facebook/sam-vit-huge"
description_model = "nvidia/DAM-3B-Self-Contained"
clip_feature_model = "OpenAI/clip-vit-large-patch14"

[processing]
batch_size = 1
frame_interval = 10
default_mode = "list | detect | segment | describe | extract_scene | extract_object"

[detection_settings]
box_threshold = 0.25
text_threshold = 0.25

[description_settings]
temperature = 0.1
top_p = 0.3
num_beams = 3
max_tokens = 1024

[visualization]
enabled = true
annotated_image_format = "png"
contour_enabled = true
contour_thickness = 5
description_viz_enabled = true
```

## Loading Configuration

### Using ConfigManager

```python
from cortexia.core.config.manager import ConfigManager

# Load configuration from file
config_manager = ConfigManager(config_file_path="config/example_config.toml")
config_manager.load_config()

# Access configuration parameters
model_name = config_manager.get_param("model_settings.object_listing_model")
batch_size = config_manager.get_param("processing.batch_size", 4)  # with default
```

### Environment Variable Override

```python
import os
from cortexia.core.config.manager import ConfigManager

# Set environment variables
os.environ["PROCESSING_BATCH_SIZE"] = "8"
os.environ["MODEL_SETTINGS_OBJECT_LISTING_MODEL"] = "custom/model"

# Load configuration (environment variables override file settings)
config_manager = ConfigManager("config/example_config.toml")
config_manager.load_config()

# Environment variables are automatically applied
batch_size = config_manager.get_param("processing.batch_size")  # Will be 8
```

### Programmatic Configuration

```python
from cortexia.core.config.manager import ConfigManager

# Create configuration programmatically
config_manager = ConfigManager()

# Set parameters directly
config_manager.set_param("processing.batch_size", 16)
config_manager.set_param("model_settings.object_detection_model", "custom/model")
config_manager.set_param("logging.level", "DEBUG")

# Save configuration to file
config_manager.save_config("config/custom_config.toml")
```

## Configuration Parameters

### Logging Configuration

```toml
[logging]
level = "INFO"          # DEBUG, INFO, WARNING, ERROR, CRITICAL
file = "cortexia.log"   # Log file path (empty for console only)
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
max_size = "10MB"       # Maximum log file size
backup_count = 5        # Number of backup log files
```

### Model Settings

```toml
[model_settings]
# Vision Language Models
object_listing_model = "Qwen/Qwen2.5-VL-3B-Instruct"
description_model = "nvidia/DAM-3B-Self-Contained"

# Object Detection
object_detection_model = "IDEA-Research/grounding-dino-base"

# Segmentation
segmentation_model = "facebook/sam-vit-huge"

# Feature Extraction
clip_feature_model = "PE-Core-B16-224"

# Captioning
image_captioning_model = "vikhyatk/moondream2"

# Depth Estimation
depth_estimation_model = "apple/DepthPro"

# Model cache settings
model_cache_dir = "/path/to/model/cache"
download_timeout = 300  # seconds
```

### Processing Settings

```toml
[processing]
# General processing
default_mode = "list | detect | segment | describe"
input_video_path = "sample_data/"
output_directory = "output/"
frame_interval = 30
batch_size = 4

# File handling
image_format = "jpg"
image_quality = 95
save_intermediate = false
cleanup_temp_files = true

# Performance settings
num_workers = 4
prefetch_factor = 2
pin_memory = true
```

### Detection Settings

```toml
[detection_settings]
# Threshold settings
box_threshold = 0.3
text_threshold = 0.3
nms_threshold = 0.5

# Model-specific settings
max_detections = 100
min_confidence = 0.1
target_size = [640, 640]

# Class filtering
include_classes = ["person", "car", "dog"]
exclude_classes = ["background"]
```

### Description Settings

```toml
[description_settings]
# Generation parameters
temperature = 0.2
top_p = 0.5
top_k = 50
num_beams = 1
max_tokens = 512

# Content control
min_length = 10
max_length = 500
do_sample = true
repetition_penalty = 1.1

# Prompt engineering
system_prompt = "Describe this image in detail:"
user_prompt_template = "Image: {image}\nDescription:"
```

### Visualization Settings

```toml
[visualization]
# General settings
enabled = true
annotated_image_format = "jpg"
image_quality = 90

# Bounding boxes
bbox_enabled = true
bbox_thickness = 2
bbox_color = [255, 0, 0]  # RGB
bbox_alpha = 0.7

# Segmentation masks
segmentation_enabled = true
contour_enabled = true
contour_thickness = 3
contour_color = [0, 255, 0]

# Text annotations
text_enabled = true
font_size = 12
font_color = [255, 255, 255]
background_color = [0, 0, 0]
background_alpha = 0.7

# Description visualization
description_viz_enabled = false
description_position = "bottom_left"
description_max_length = 100
```

## Environment Variables

### Supported Environment Variables

```bash
# Logging
CORTEXIA_LOG_LEVEL=DEBUG
CORTEXIA_LOG_FILE=/path/to/logfile.log

# Model Settings
CORTEXIA_MODEL_SETTINGS_OBJECT_LISTING_MODEL=custom/model
CORTEXIA_MODEL_SETTINGS_OBJECT_DETECTION_MODEL=custom/detector

# Processing
CORTEXIA_PROCESSING_BATCH_SIZE=8
CORTEXIA_PROCESSING_FRAME_INTERVAL=15
CORTEXIA_PROCESSING_INPUT_VIDEO_PATH=/path/to/videos
CORTEXIA_PROCESSING_OUTPUT_DIRECTORY=/path/to/output

# Detection
CORTEXIA_DETECTION_SETTINGS_BOX_THRESHOLD=0.4
CORTEXIA_DETECTION_SETTINGS_TEXT_THRESHOLD=0.4

# Description
CORTEXIA_DESCRIPTION_SETTINGS_TEMPERATURE=0.1
CORTEXIA_DESCRIPTION_SETTINGS_MAX_TOKENS=256

# Model Cache
HF_HOME=/path/to/huggingface/cache
TRANSFORMERS_CACHE=/path/to/transformers/cache
TORCH_HOME=/path/to/torch/cache
```

### Environment Variable Format

Environment variables follow the pattern:
```
CORTEXIA_{SECTION}_{PARAMETER}=value
```

Where:
- `SECTION` is the TOML section name in uppercase
- `PARAMETER` is the parameter name in uppercase
- Nested parameters use underscores instead of dots

Examples:
- `model_settings.object_listing_model` → `CORTEXIA_MODEL_SETTINGS_OBJECT_LISTING_MODEL`
- `detection_settings.box_threshold` → `CORTEXIA_DETECTION_SETTINGS_BOX_THRESHOLD`

## Configuration Validation

### Built-in Validation

```python
from cortexia.core.config.manager import ConfigManager

try:
    config_manager = ConfigManager("config/example_config.toml")
    config_manager.load_config()
    print("✓ Configuration is valid")
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

### Custom Validation

```python
from cortexia.core.config.manager import ConfigManager
from cortexia.core.config.schemas import validate_config

def validate_custom_config(config_manager):
    """Custom validation logic."""
    
    # Check batch size is reasonable
    batch_size = config_manager.get_param("processing.batch_size")
    if batch_size > 32:
        raise ValueError("Batch size too large for typical GPUs")
    
    # Check model compatibility
    detection_model = config_manager.get_param("model_settings.object_detection_model")
    if "grounding-dino" in detection_model and batch_size > 1:
        print("Warning: Grounding DINO works best with batch_size=1")
    
    # Check paths exist
    input_path = config_manager.get_param("processing.input_video_path")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    return True

# Usage
config_manager = ConfigManager("config/example_config.toml")
config_manager.load_config()
validate_custom_config(config_manager)
```

## Configuration Templates

### Development Configuration

```toml
[logging]
level = "DEBUG"
file = "dev.log"

[processing]
batch_size = 2
frame_interval = 10
default_mode = "list | detect | segment"

[model_settings]
object_detection_model = "YOLO-World/small"
segmentation_model = "facebook/sam-vit-base"

[visualization]
enabled = true
description_viz_enabled = true
```

### Production Configuration

```toml
[logging]
level = "WARNING"
file = "/var/log/cortexia/production.log"
max_size = "100MB"
backup_count = 10

[processing]
batch_size = 16
frame_interval = 30
default_mode = "list | detect | segment | describe"
num_workers = 8

[model_settings]
object_listing_model = "Qwen/Qwen2.5-VL-7B-Instruct"
object_detection_model = "IDEA-Research/grounding-dino-large"

[visualization]
enabled = false  # Disable for performance
```

### Research Configuration

```toml
[logging]
level = "INFO"
file = "research.log"

[processing]
batch_size = 1
frame_interval = 1
default_mode = "list | detect | segment | describe | extract_scene | extract_object"
save_intermediate = true

[model_settings]
object_listing_model = "Qwen/Qwen2.5-VL-7B-Instruct"
object_detection_model = "IDEA-Research/grounding-dino-large"
segmentation_model = "facebook/sam-vit-huge"
description_model = "nvidia/DAM-3B-Self-Contained"

[detection_settings]
box_threshold = 0.25
text_threshold = 0.25
max_detections = 500

[description_settings]
temperature = 0.1
top_p = 0.3
num_beams = 5
max_tokens = 1024

[visualization]
enabled = true
annotated_image_format = "png"
image_quality = 100
```

## Configuration Best Practices

### 1. Environment-Specific Configurations

```bash
# Directory structure
config/
├── base.toml              # Common settings
├── development.toml       # Development overrides
├── production.toml        # Production overrides
└── testing.toml          # Testing overrides
```

```python
# Load environment-specific configuration
import os
from cortexia.core.config.manager import ConfigManager

env = os.environ.get("CORTEXIA_ENV", "development")
config_files = [
    "config/base.toml",
    f"config/{env}.toml"
]

config_manager = ConfigManager()
for config_file in config_files:
    if os.path.exists(config_file):
        config_manager.load_config(config_file)
```

### 2. Configuration Validation Pipeline

```python
def validate_configuration_pipeline(config_manager):
    """Complete configuration validation pipeline."""
    
    # 1. Basic validation
    config_manager.validate()
    
    # 2. Resource validation
    validate_system_resources(config_manager)
    
    # 3. Model compatibility validation
    validate_model_compatibility(config_manager)
    
    # 4. Path validation
    validate_paths(config_manager)
    
    # 5. Performance validation
    validate_performance_settings(config_manager)
    
    return True
```

### 3. Configuration Backup and Versioning

```python
import shutil
from datetime import datetime

def backup_configuration(config_file):
    """Create a backup of configuration file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{config_file}.backup_{timestamp}"
    shutil.copy2(config_file, backup_file)
    return backup_file

def version_configuration(config_file, version):
    """Create a versioned copy of configuration."""
    versioned_file = f"{config_file}.v{version}"
    shutil.copy2(config_file, versioned_file)
    return versioned_file
```

### 4. Configuration Monitoring

```python
import time
import hashlib
from pathlib import Path

def monitor_configuration(config_file, callback=None):
    """Monitor configuration file for changes."""
    
    def get_file_hash(filepath):
        """Get file hash for change detection."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    last_hash = get_file_hash(config_file)
    
    while True:
        time.sleep(5)  # Check every 5 seconds
        current_hash = get_file_hash(config_file)
        
        if current_hash != last_hash:
            last_hash = current_hash
            if callback:
                callback(config_file)
            print(f"Configuration file changed: {config_file}")
```

## Troubleshooting Configuration Issues

### Common Issues and Solutions

#### 1. Configuration File Not Found

```python
# Check if configuration file exists
import os
config_path = "config/example_config.toml"

if not os.path.exists(config_path):
    print(f"Configuration file not found: {config_path}")
    print("Available configuration files:")
    for file in os.listdir("config/"):
        if file.endswith(".toml"):
            print(f"  - config/{file}")
```

#### 2. Invalid Configuration Values

```python
# Validate configuration values
from cortexia.core.config.manager import ConfigManager

try:
    config_manager = ConfigManager("config/example_config.toml")
    config_manager.load_config()
    
    # Check specific values
    batch_size = config_manager.get_param("processing.batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
        
except Exception as e:
    print(f"Configuration validation failed: {e}")
```

#### 3. Model Loading Issues

```python
# Test model loading with current configuration
def test_model_loading(config_manager):
    """Test if all configured models can be loaded."""
    
    models_to_test = [
        "model_settings.object_listing_model",
        "model_settings.object_detection_model",
        "model_settings.segmentation_model"
    ]
    
    for model_param in models_to_test:
        model_name = config_manager.get_param(model_param)
        try:
            # Attempt to load model (simplified test)
            print(f"Testing model: {model_name}")
            # Add actual model loading test here
            print(f"✓ {model_name} - OK")
        except Exception as e:
            print(f"✗ {model_name} - Failed: {e}")
```

#### 4. Performance Issues

```python
# Analyze configuration for performance bottlenecks
def analyze_performance_config(config_manager):
    """Analyze configuration for potential performance issues."""
    
    batch_size = config_manager.get_param("processing.batch_size")
    frame_interval = config_manager.get_param("processing.frame_interval")
    num_workers = config_manager.get_param("processing.num_workers", 1)
    
    warnings = []
    
    if batch_size > 8:
        warnings.append("Large batch size may cause memory issues")
    
    if frame_interval < 5:
        warnings.append("Small frame interval may slow processing")
    
    if num_workers > 8:
        warnings.append("Too many workers may cause contention")
    
    return warnings
```

## Next Steps

After configuring the SDK:

1. **Test Configuration**: Run validation tests to ensure configuration is correct
2. **Process Sample Data**: Test with sample videos to verify configuration works
3. **Optimize Performance**: Adjust parameters based on your hardware and use case
4. **Monitor Performance**: Use logging and monitoring to track performance

For more information, see:

- [Getting Started](getting_started.md) for basic usage
- [Installation](installation.md) for system requirements
- [Troubleshooting](troubleshooting.md) for common issues