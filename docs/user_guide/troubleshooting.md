# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the Cortexia Video SDK. It covers installation problems, runtime errors, performance issues, and debugging techniques.

## Quick Reference

| Issue Category | Common Symptoms | Quick Fix |
|----------------|------------------|-----------|
| **Installation** | Import errors, missing dependencies | `pip install -e .` |
| **Model Loading** | Timeout errors, cache issues | Check `HF_HOME` and permissions |
| **Memory Issues** | OOM errors, slow processing | Reduce `batch_size`, use CPU |
| **Configuration** | Invalid config, missing files | Validate TOML syntax |
| **Performance** | Slow processing, high latency | Optimize batch size and workers |

## Installation Issues

### Problem: Import Errors

**Symptoms:**
```bash
ImportError: No module named 'cortexia'
ModuleNotFoundError: No module named 'cortexia'
```

**Solutions:**

1. **Check Installation:**
```bash
# Verify installation
pip list | grep cortexia

# If not found, install in development mode
pip install -e .
```

2. **Check Python Version:**
```bash
# Must be Python 3.10+
python --version

# If wrong version, use correct Python
python3.10 -m pip install -e .
```

3. **Check Virtual Environment:**
```bash
# Verify virtual environment is active
echo $VIRTUAL_ENV

# If not active, activate it
source venv/bin/activate  # Linux/Mac
# or
source venv/Scripts/activate  # Windows
```

4. **Check PATH:**
```bash
# Verify Python PATH
which python
which pip

# Should point to virtual environment
```

### Problem: Missing Dependencies

**Symptoms:**
```bash
ImportError: No module named 'torch'
ImportError: No module named 'transformers'
```

**Solutions:**

1. **Install Dependencies:**
```bash
# Install all dependencies
pip install -e .

# Or install specific dependencies
pip install torch torchvision torchaudio
pip install transformers
pip install opencv-python
```

2. **Check Requirements:**
```bash
# Install from requirements
pip install -r requirements.txt

# Or use uv (recommended)
uv sync
```

3. **Verify CUDA Compatibility:**
```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Model Loading Issues

### Problem: Model Download Timeout

**Symptoms:**
```bash
TimeoutError: Request timed out
ConnectionError: Failed to establish connection
```

**Solutions:**

1. **Check Internet Connection:**
```bash
# Test connectivity
ping huggingface.co
curl -I https://huggingface.co
```

2. **Set Custom Cache Directory:**
```bash
# Set environment variables
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
export TORCH_HOME=/path/to/cache

# Or in Python
import os
os.environ["HF_HOME"] = "/path/to/cache"
```

3. **Increase Timeout:**
```python
import cortexia
from cortexia.core.config.manager import ConfigManager

# Set longer timeout
config_manager = ConfigManager()
config_manager.set_param("model_settings.download_timeout", 600)
```

4. **Use Local Models:**
```python
# Download models first
from huggingface_hub import snapshot_download
snapshot_download("IDEA-Research/grounding-dino-base")

# Then use local path
config_manager.set_param("model_settings.object_detection_model", "/path/to/local/model")
```

### Problem: Model Cache Permissions

**Symptoms:**
```bash
PermissionError: [Errno 13] Permission denied
OSError: [Errno 28] No space left on device
```

**Solutions:**

1. **Check Cache Directory Permissions:**
```bash
# Check permissions
ls -la ~/.cache/huggingface/
ls -la ~/.cache/torch/

# Fix permissions
chmod -R 755 ~/.cache/huggingface/
chmod -R 755 ~/.cache/torch/
```

2. **Check Disk Space:**
```bash
# Check available space
df -h
du -sh ~/.cache/huggingface/

# Clean up if needed
rm -rf ~/.cache/huggingface/hub/models--unused-model
```

3. **Use Alternative Cache Location:**
```bash
# Create cache directory with proper permissions
mkdir -p /tmp/huggingface_cache
chmod 777 /tmp/huggingface_cache

# Set environment variable
export HF_HOME=/tmp/huggingface_cache
```

## Memory Issues

### Problem: Out of Memory (OOM) Errors

**Symptoms:**
```bash
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Reduce Batch Size:**
```python
from cortexia.core.config.manager import ConfigManager

config_manager = ConfigManager()
config_manager.set_param("processing.batch_size", 1)
```

2. **Use CPU Instead of GPU:**
```python
import torch

# Force CPU usage
torch.set_num_threads(1)
device = torch.device("cpu")

# Or disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

3. **Enable Memory Optimization:**
```python
# Enable gradient checkpointing
config_manager.set_param("model_settings.gradient_checkpointing", True)

# Use mixed precision
config_manager.set_param("model_settings.mixed_precision", True)
```

4. **Clear GPU Cache:**
```python
import torch
import gc

# Clear cache
torch.cuda.empty_cache()
gc.collect()

# Monitor memory
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Problem: Memory Leaks

**Symptoms:**
```bash
Memory usage increases over time
System becomes unresponsive
```

**Solutions:**

1. **Monitor Memory Usage:**
```python
import psutil
import torch

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"RSS Memory: {memory_info.rss / 1e9:.2f} GB")
    print(f"VMS Memory: {memory_info.vms / 1e9:.2f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

2. **Implement Memory Cleanup:**
```python
def process_with_cleanup(frame_packets):
    try:
        # Process frames
        results = detector.process_batch(frame_packets)
        return results
    finally:
        # Cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

3. **Use Context Managers:**
```python
from contextlib import contextmanager

@contextmanager
def memory_context():
    try:
        yield
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage
with memory_context():
    results = detector.process_batch(frame_packets)
```

## Configuration Issues

### Problem: Invalid Configuration File

**Symptoms:**
```bash
toml.TomlDecodeError: Invalid TOML
ValueError: Invalid configuration value
```

**Solutions:**

1. **Validate TOML Syntax:**
```bash
# Use TOML linter
pip install toml-cli
toml-cli validate config/example_config.toml

# Or use Python
import toml
try:
    with open("config/example_config.toml", "r") as f:
        config = toml.load(f)
    print("✓ TOML syntax is valid")
except Exception as e:
    print(f"✗ TOML syntax error: {e}")
```

2. **Check Configuration Values:**
```python
from cortexia.core.config.manager import ConfigManager

def validate_config(config_file):
    """Validate configuration file."""
    try:
        config_manager = ConfigManager(config_file)
        config_manager.load_config()
        
        # Check required parameters
        required_params = [
            "model_settings.object_detection_model",
            "processing.batch_size",
            "processing.frame_interval"
        ]
        
        for param in required_params:
            value = config_manager.get_param(param)
            if value is None:
                print(f"✗ Missing required parameter: {param}")
                return False
        
        print("✓ Configuration is valid")
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
```

3. **Use Configuration Templates:**
```python
# Start with a known good configuration
config_manager = ConfigManager("config/example_config.toml")

# Modify specific parameters
config_manager.set_param("processing.batch_size", 8)
config_manager.set_param("model_settings.object_detection_model", "custom/model")

# Save new configuration
config_manager.save_config("config/custom_config.toml")
```

### Problem: Environment Variable Issues

**Symptoms:**
```bash
Environment variables not being applied
Configuration values not overridden
```

**Solutions:**

1. **Check Environment Variables:**
```bash
# List all Cortexia environment variables
env | grep CORTEXIA

# Check specific variable
echo $CORTEXIA_PROCESSING_BATCH_SIZE
```

2. **Set Environment Variables Correctly:**
```bash
# Set in shell
export CORTEXIA_PROCESSING_BATCH_SIZE=8
export CORTEXIA_MODEL_SETTINGS_OBJECT_DETECTION_MODEL="custom/model"

# Set in Python
import os
os.environ["CORTEXIA_PROCESSING_BATCH_SIZE"] = "8"
os.environ["CORTEXIA_MODEL_SETTINGS_OBJECT_DETECTION_MODEL"] = "custom/model"
```

3. **Debug Environment Variable Loading:**
```python
from cortexia.core.config.manager import ConfigManager

def debug_env_vars():
    """Debug environment variable loading."""
    
    # Set test environment variable
    os.environ["CORTEXIA_PROCESSING_BATCH_SIZE"] = "16"
    
    # Load configuration
    config_manager = ConfigManager("config/example_config.toml")
    config_manager.load_config()
    
    # Check if environment variable was applied
    batch_size = config_manager.get_param("processing.batch_size")
    print(f"Batch size from config: {batch_size}")
    
    # Should be 16 (from environment variable)
    if batch_size == 16:
        print("✓ Environment variables working correctly")
    else:
        print("✗ Environment variables not working")
```

## Performance Issues

### Problem: Slow Processing

**Symptoms:**
```bash
Processing takes too long
High latency between frames
```

**Solutions:**

1. **Optimize Batch Size:**
```python
def find_optimal_batch_size():
    """Find optimal batch size for your hardware."""
    
    batch_sizes = [1, 2, 4, 8, 16]
    frame_packets = [create_test_frame() for _ in range(32)]
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(frame_packets), batch_size):
            batch = frame_packets[i:i + batch_size]
            detector.process_batch(batch)
        
        elapsed = time.time() - start_time
        fps = len(frame_packets) / elapsed
        
        print(f"Batch size {batch_size}: {fps:.2f} FPS")
```

2. **Enable Parallel Processing:**
```python
# Configure workers
config_manager.set_param("processing.num_workers", 4)
config_manager.set_param("processing.prefetch_factor", 2)
config_manager.set_param("processing.pin_memory", True)
```

3. **Use Appropriate Models:**
```python
# Use smaller models for faster processing
config_manager.set_param("model_settings.object_detection_model", "YOLO-World/small")
config_manager.set_param("model_settings.segmentation_model", "facebook/sam-vit-base")
```

4. **Optimize Frame Processing:**
```python
# Reduce frame resolution
def resize_frame(frame_packet, target_size=(640, 640)):
    """Resize frame for faster processing."""
    import cv2
    
    resized = cv2.resize(frame_packet.frame_data, target_size)
    return VideoFramePacket(
        frame_data=resized,
        frame_number=frame_packet.frame_number,
        timestamp=frame_packet.timestamp,
        source_video_id=frame_packet.source_video_id
    )
```

### Problem: High CPU Usage

**Symptoms:**
```bash
CPU usage consistently at 100%
System becomes unresponsive
```

**Solutions:**

1. **Limit CPU Threads:**
```python
import torch

# Limit PyTorch threads
torch.set_num_threads(4)

# Limit OpenCV threads
import cv2
cv2.setNumThreads(2)
```

2. **Use GPU Acceleration:**
```python
# Enable CUDA if available
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("Using CPU")
```

3. **Implement Rate Limiting:**
```python
import time
from threading import Lock

class RateLimiter:
    def __init__(self, max_calls_per_second):
        self.max_calls = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second
        self.last_call = 0
        self.lock = Lock()
    
    def wait(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call
            
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            
            self.last_call = time.time()

# Usage
rate_limiter = RateLimiter(max_calls_per_second=10)

for frame in frame_packets:
    rate_limiter.wait()
    result = detector.process_frame(frame)
```

## Runtime Errors

### Problem: Component Creation Failed

**Symptoms:**
```bash
ValueError: Unknown component type
KeyError: Component not found in registry
```

**Solutions:**

1. **List Available Components:**
```python
import cortexia

# List all available components
features = cortexia.list_features()
gates = cortexia.list_gates()
samplers = cortexia.list_samplers()

print("Available features:", features)
print("Available gates:", gates)
print("Available samplers:", samplers)
```

2. **Check Component Registry:**
```python
from cortexia.core.registry import get_feature_hub, get_gate_hub

# Check feature registry
feature_hub = get_feature_hub()
print("Registered features:", list(feature_hub.keys()))

# Check gate registry
gate_hub = get_gate_hub()
print("Registered gates:", list(gate_hub.keys()))
```

3. **Register Custom Components:**
```python
from cortexia.core.registry import get_feature_hub

# Get registry
FEATURE_REGISTRY = get_feature_hub()

# Register custom component
@FEATURE_REGISTRY.decorator("my_custom_feature")
class CustomFeature:
    def __init__(self, config):
        self.config = config
    
    def process_frame(self, frame_packet):
        # Custom processing logic
        pass

# Now create component
custom_feature = cortexia.create_feature("my_custom_feature")
```

### Problem: Frame Processing Errors

**Symptoms:**
```bash
ValueError: Invalid frame format
TypeError: Expected VideoFramePacket, got numpy.ndarray
AttributeError: 'NoneType' object has no attribute 'shape'
```

**Solutions:**

1. **Validate Frame Data:**
```python
def validate_frame_packet(frame_packet):
    """Validate frame packet structure."""
    
    # Check if it's the right type
    if not hasattr(frame_packet, 'frame_data'):
        raise ValueError("Frame packet must have 'frame_data' attribute")
    
    # Check frame data
    if frame_packet.frame_data is None:
        raise ValueError("Frame data cannot be None")
    
    # Check frame dimensions
    if len(frame_packet.frame_data.shape) != 3:
        raise ValueError("Frame data must be 3-dimensional (H, W, C)")
    
    # Check required attributes
    required_attrs = ['frame_number', 'timestamp', 'source_video_id']
    for attr in required_attrs:
        if not hasattr(frame_packet, attr):
            raise ValueError(f"Frame packet missing required attribute: {attr}")
    
    return True
```

2. **Convert Frame Data:**
```python
import numpy as np
import cv2
from cortexia.data.models.video import VideoFramePacket

def convert_to_frame_packet(frame_data, frame_number=0, timestamp=0.0, source_video_id="unknown"):
    """Convert various frame formats to VideoFramePacket."""
    
    # Handle numpy array
    if isinstance(frame_data, np.ndarray):
        # Convert BGR to RGB if needed
        if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
            # Assume BGR format, convert to RGB
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        
        return VideoFramePacket(
            frame_data=frame_data,
            frame_number=frame_number,
            timestamp=timestamp,
            source_video_id=source_video_id
        )
    
    # Handle other formats as needed
    raise ValueError(f"Unsupported frame data type: {type(frame_data)}")
```

3. **Debug Frame Processing:**
```python
def debug_frame_processing(frame_packet):
    """Debug frame processing pipeline."""
    
    print("=== Frame Processing Debug ===")
    print(f"Frame number: {frame_packet.frame_number}")
    print(f"Timestamp: {frame_packet.timestamp}")
    print(f"Source video: {frame_packet.source_video_id}")
    print(f"Frame shape: {frame_packet.frame_data.shape}")
    print(f"Frame dtype: {frame_packet.frame_data.dtype}")
    print(f"Frame min/max: {frame_packet.frame_data.min()}/{frame_packet.frame_data.max()}")
    
    # Test with a simple gate
    try:
        blur_gate = cortexia.create_gate("blur")
        result = blur_gate.process_frame(frame_packet)
        print(f"Blur gate result: {result.passed} (score: {result.score})")
    except Exception as e:
        print(f"Blur gate failed: {e}")
    
    # Test with a simple feature
    try:
        lister = cortexia.create_feature("listing")
        result = lister.process_frame(frame_packet)
        print(f"Listing result: {len(result.objects)} objects found")
    except Exception as e:
        print(f"Listing feature failed: {e}")
```

## Debugging Techniques

### Enable Debug Logging

```python
import logging
from cortexia.core.config.manager import ConfigManager

# Enable debug logging
config_manager = ConfigManager()
config_manager.set_param("logging.level", "DEBUG")

# Or set directly
logging.basicConfig(level=logging.DEBUG)

# Create logger
logger = logging.getLogger("cortexia")
logger.setLevel(logging.DEBUG)
```

### Use Debug Mode

```python
import cortexia

# Enable debug mode
cortexia.set_debug_mode(True)

# Now all operations will produce debug output
detector = cortexia.create_feature("detection")
result = detector.process_frame(frame_packet)
```

### Performance Profiling

```python
import cProfile
import pstats
import time
from contextlib import contextmanager

@contextmanager
def profile_context(name):
    """Context manager for profiling."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    
    profiler.disable()
    
    print(f"=== {name} Performance ===")
    print(f"Total time: {elapsed:.2f} seconds")
    
    # Print top 10 functions by time
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

# Usage
with profile_context("Frame Processing"):
    result = detector.process_frame(frame_packet)
```

### Memory Profiling

```python
import tracemalloc

def profile_memory():
    """Profile memory usage."""
    
    # Start tracing
    tracemalloc.start()
    
    # Process frames
    results = []
    for frame in frame_packets:
        result = detector.process_frame(frame)
        results.append(result)
    
    # Get memory statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory usage: {current / 1e6:.2f} MB")
    print(f"Peak memory usage: {peak / 1e6:.2f} MB")
    
    return results
```

## Getting Help

### Diagnostic Script

```python
#!/usr/bin/env python3
"""
Cortexia Video SDK Diagnostic Script
Run this script to diagnose common issues.
"""

import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("=== Python Version ===")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("✗ Python 3.10+ required")
        return False
    else:
        print("✓ Python version OK")
        return True

def check_dependencies():
    """Check required dependencies."""
    print("\n=== Dependencies ===")
    
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('toml', 'toml'),
        ('PIL', 'PIL'),
    ]
    
    all_ok = True
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - missing")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability."""
    print("\n=== CUDA Status ===")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("✗ CUDA not available")
        return False

def check_cortexia_installation():
    """Check Cortexia installation."""
    print("\n=== Cortexia Installation ===")
    
    try:
        import cortexia
        print(f"✓ Cortexia installed")
        print(f"  Version: {getattr(cortexia, '__version__', 'unknown')}")
        
        # Test component creation
        features = cortexia.list_features()
        gates = cortexia.list_gates()
        print(f"  Features: {len(features)} available")
        print(f"  Gates: {len(gates)} available")
        
        return True
    except ImportError as e:
        print(f"✗ Cortexia not installed: {e}")
        return False

def check_configuration():
    """Check configuration files."""
    print("\n=== Configuration ===")
    
    config_files = [
        "config/example_config.toml",
        "config/light_mode.toml",
        "config/heavy_mode.toml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file}")
        else:
            print(f"✗ {config_file} - missing")
    
    return True

def check_model_cache():
    """Check model cache directory."""
    print("\n=== Model Cache ===")
    
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
        os.environ.get("HF_HOME", "")
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir and Path(cache_dir).exists():
            size = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file())
            print(f"✓ {cache_dir} ({size / 1e9:.2f} GB)")
        else:
            print(f"✗ {cache_dir} - missing or empty")
    
    return True

def run_basic_test():
    """Run a basic functionality test."""
    print("\n=== Basic Test ===")
    
    try:
        import cortexia
        from cortexia.data.models.video import VideoFramePacket
        
        # Create test frame
        frame_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frame_packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=0.0,
            source_video_id="test"
        )
        
        # Test component creation
        blur_gate = cortexia.create_gate("blur")
        result = blur_gate.process_frame(frame_packet)
        
        print(f"✓ Basic test passed")
        print(f"  Blur gate result: {result.passed}")
        
        return True
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False

def main():
    """Run all diagnostic checks."""
    print("Cortexia Video SDK Diagnostic Tool")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_cuda,
        check_cortexia_installation,
        check_configuration,
        check_model_cache,
        run_basic_test
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"✗ Check failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("=== Summary ===")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed - see details above")
        print("\nNext steps:")
        print("1. Check the failed sections above")
        print("2. Follow the troubleshooting guide")
        print("3. Check the documentation")
        print("4. Create an issue if problems persist")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### When to Create an Issue

Create a GitHub issue if:

1. **You've tried all troubleshooting steps** and the problem persists
2. **You encounter a bug** that's not covered in this guide
3. **You have a feature request** or improvement suggestion
4. **The documentation is unclear** or incomplete

### Issue Template

```markdown
## Issue Description
[Describe the problem you're experiencing]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [Third step]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Environment
- **OS**: [e.g., Ubuntu 20.04, macOS 13.0, Windows 11]
- **Python**: [e.g., 3.10.6]
- **Cortexia Version**: [e.g., 0.1.0]
- **CUDA Version**: [e.g., 11.8]
- **GPU**: [e.g., RTX 3090]

## Error Messages
```
[Paste error messages here]
```

## Additional Context
[Any additional information that might be helpful]
```

## Next Steps

After resolving your issue:

1. **Document the solution**: Add notes to your personal documentation
2. **Share with others**: Help others in the community
3. **Contribute improvements**: Suggest improvements to the documentation
4. **Monitor performance**: Keep an eye on system performance

Remember that troubleshooting is a normal part of working with complex systems. Be patient and systematic in your approach! 🛠️