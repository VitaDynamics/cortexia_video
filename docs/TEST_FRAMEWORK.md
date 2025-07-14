## 15. Test Framework Design

### 15.1 Overview

The cortexia_video project requires a comprehensive test framework that addresses the unique challenges of testing ML-heavy video processing systems. The framework implements a three-tier testing strategy that separates lightweight unit tests from GPU-intensive integration tests, enabling fast development cycles while ensuring production reliability.

### 15.2 Testing Paradigm

**Core Philosophy**: Separate tests by computational requirements and execution speed to optimize developer productivity and CI/CD efficiency.

**Three-Tier Strategy**:
1. **Lightweight Tests** (CPU-only, <1s execution): Unit tests for business logic, configuration, utilities
2. **Mock-based Tests** (CPU-only, <10s execution): Integration tests using dummy models via factory pattern
3. **GPU-heavy Tests** (GPU-required, >30s execution): Full integration tests with real ML models

### 15.3 Test Categorization Strategy

#### 15.3.1 Lightweight Tests (Fast, CPU-only)
**Target Components**:
- [`ConfigManager`](cortexia_video/config_manager.py:12): Configuration loading, parameter validation, TOML/JSON parsing
- Data models in [`schemes.py`](cortexia_video/schemes.py:8): Pydantic model validation, serialization/deserialization
- Utility functions in [`utils.py`](cortexia_video/utils.py:6): Image cropping, polygon operations, mask conversions
- Parser functions in [`parser.py`](cortexia_video/parser.py:4): String parsing, list formatting
- Data I/O functions in [`data_io.py`](cortexia_video/data_io.py:12): File operations (mocked), JSON serialization

**Characteristics**:
- No model downloads or GPU operations
- Execution time: <1 second per test
- Can run on any development machine
- Ideal for TDD and rapid iteration

#### 15.3.2 GPU-heavy Tests (Slow, GPU-required)
**Target Components**:
- [`ObjectDetector`](cortexia_video/object_detection.py:11): GroundingDINO model inference
- [`ObjectSegmenter`](cortexia_video/object_segmentation.py:14): SAM model segmentation
- [`CLIPFeatureExtractor`](cortexia_video/feature_extraction.py:105): PE-Core CLIP feature extraction
- [`DepthProEstimator`](cortexia_video/depth_estimation.py:29): DepthPro depth estimation
- VLM object listers: Qwen2.5VL, RAM model inference
- [`ProcessingManager`](cortexia_video/processing_manager.py:28): End-to-end pipeline integration

**Characteristics**:
- Requires GPU hardware and model downloads
- Execution time: 30+ seconds per test
- Tests real model behavior and performance
- Critical for production validation

#### 15.3.3 Mock-based Tests (Medium speed, CPU-only)
**Target Components**:
- ML component interfaces using factory pattern with dummy models
- [`ProcessingManager`](cortexia_video/processing_manager.py:28) orchestration logic with mocked components
- Pipeline integration testing with synthetic data
- Error handling and edge case scenarios

**Characteristics**:
- Uses dummy models that mimic real model interfaces
- Execution time: 1-10 seconds per test
- Tests business logic without GPU dependencies
- Enables comprehensive testing on CPU-only systems

### 15.4 Decorator System Implementation

#### 15.4.1 Core Decorators

```python
# tests/test_decorators.py
import os
import pytest
import torch
from functools import wraps

def require_torch_gpu(func):
    """
    Decorator to skip tests that require GPU if no GPU is available or SKIP_GPU is set.
    
    Usage:
        @require_torch_gpu
        def test_gpu_inference(self):
            # Test code that requires GPU
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv('SKIP_GPU', '').lower() in ('true', '1', 'yes'):
            pytest.skip('GPU tests skipped due to SKIP_GPU environment variable')
        if not torch.cuda.is_available():
            pytest.skip('GPU not available')
        return func(*args, **kwargs)
    return pytest.mark.gpu(wrapper)

def slow(func):
    """
    Decorator to mark slow tests that can be skipped unless RUN_SLOW is set.
    
    Usage:
        @slow
        def test_full_pipeline(self):
            # Long-running integration test
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv('RUN_SLOW', '').lower() not in ('true', '1', 'yes'):
            pytest.skip('Slow tests skipped unless RUN_SLOW=true')
        return func(*args, **kwargs)
    return pytest.mark.slow(wrapper)

def require_model_download(func):
    """
    Decorator for tests that require downloading large model weights.
    
    Usage:
        @require_model_download
        def test_real_model_loading(self):
            # Test that downloads model weights
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv('SKIP_MODEL_DOWNLOAD', '').lower() in ('true', '1', 'yes'):
            pytest.skip('Model download tests skipped due to SKIP_MODEL_DOWNLOAD environment variable')
        return func(*args, **kwargs)
    return pytest.mark.model_download(wrapper)
```

#### 15.4.2 Environment Variable Controls

| Variable | Values | Effect |
|----------|--------|--------|
| `SKIP_GPU` | `true`, `1`, `yes` | Skip all GPU-requiring tests |
| `RUN_SLOW` | `true`, `1`, `yes` | Enable slow/integration tests |
| `USE_DUMMY_MODELS` | `true`, `1`, `yes` | Force factory pattern to use dummy models |
| `SKIP_MODEL_DOWNLOAD` | `true`, `1`, `yes` | Skip tests requiring model downloads |

### 15.5 Factory Pattern for Dummy Models

#### 15.5.1 Model Factory Implementation

```python
# tests/test_factories.py
import os
import torch
import numpy as np
from typing import List, Dict, Any
from PIL import Image
from cortexia_video.config_manager import ConfigManager

class ModelFactory:
    """Factory for creating real or dummy ML models based on configuration."""
    
    @staticmethod
    def create_object_detector(config_manager: ConfigManager, use_dummy: bool = None):
        """Create ObjectDetector or DummyObjectDetector based on configuration."""
        if use_dummy is None:
            use_dummy = os.getenv('USE_DUMMY_MODELS', '').lower() in ('true', '1', 'yes')
        
        if use_dummy:
            return DummyObjectDetector(config_manager)
        else:
            from cortexia_video.object_detection import ObjectDetector
            return ObjectDetector(config_manager)
    
    @staticmethod
    def create_object_segmenter(config_manager: ConfigManager, use_dummy: bool = None):
        """Create ObjectSegmenter or DummyObjectSegmenter based on configuration."""
        if use_dummy is None:
            use_dummy = os.getenv('USE_DUMMY_MODELS', '').lower() in ('true', '1', 'yes')
        
        if use_dummy:
            return DummyObjectSegmenter(config_manager)
        else:
            from cortexia_video.object_segmentation import ObjectSegmenter
            return ObjectSegmenter(config_manager)
    
    @staticmethod
    def create_feature_extractor(config_manager: ConfigManager, use_dummy: bool = None):
        """Create CLIPFeatureExtractor or DummyCLIPFeatureExtractor based on configuration."""
        if use_dummy is None:
            use_dummy = os.getenv('USE_DUMMY_MODELS', '').lower() in ('true', '1', 'yes')
        
        if use_dummy:
            return DummyCLIPFeatureExtractor(config_manager)
        else:
            from cortexia_video.feature_extraction import CLIPFeatureExtractor
            return CLIPFeatureExtractor(config_manager)
    
    @staticmethod
    def create_depth_estimator(config_manager: ConfigManager, use_dummy: bool = None):
        """Create DepthProEstimator or DummyDepthEstimator based on configuration."""
        if use_dummy is None:
            use_dummy = os.getenv('USE_DUMMY_MODELS', '').lower() in ('true', '1', 'yes')
        
        if use_dummy:
            return DummyDepthEstimator(config_manager)
        else:
            from cortexia_video.depth_estimation import DepthProEstimator
            return DepthProEstimator()
```

#### 15.5.2 Dummy Model Implementations

```python
class DummyObjectDetector:
    """Lightweight dummy implementation of ObjectDetector for testing."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.device = "cpu"  # Always use CPU for dummy models
    
    def detect_objects(self, images_data: List[Image.Image], text_prompts: List[List[str]]) -> List[List[dict]]:
        """Return synthetic detection results without GPU computation."""
        results = []
        for i, (image, prompts) in enumerate(zip(images_data, text_prompts)):
            detections = []
            for j, prompt in enumerate(prompts[:2]):  # Limit to 2 detections per image
                detection = {
                    'score': 0.8 + (j * 0.1),  # Synthetic confidence scores
                    'label': prompt,
                    'box': [50 + j*100, 50 + j*50, 150 + j*100, 150 + j*50]  # Synthetic boxes
                }
                detections.append(detection)
            results.append(detections)
        return results

class DummyObjectSegmenter:
    """Lightweight dummy implementation of ObjectSegmenter for testing."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.device = "cpu"
    
    def segment_object(self, images_data: List[Image.Image], 
                      batch_input_boxes: List[List[List[float]]], 
                      batch_input_points=None, batch_input_labels=None) -> List[List[np.ndarray]]:
        """Return synthetic segmentation masks without GPU computation."""
        results = []
        for image, input_boxes in zip(images_data, batch_input_boxes):
            image_masks = []
            for box in input_boxes:
                # Create a simple rectangular mask based on the input box
                mask = np.zeros((image.height, image.width), dtype=bool)
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = True
                image_masks.append(mask)
            results.append(image_masks)
        return results

class DummyCLIPFeatureExtractor:
    """Lightweight dummy implementation of CLIPFeatureExtractor for testing."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.device = "cpu"
        self.feature_dim = 512  # Standard CLIP feature dimension
    
    def extract_image_features(self, images_data: List[Image.Image]) -> torch.Tensor:
        """Return synthetic normalized feature vectors."""
        if not images_data:
            return torch.empty((0, self.feature_dim))
        
        # Generate random normalized features
        features = torch.randn(len(images_data), self.feature_dim)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def extract_text_features(self, text_prompts: List[str]) -> torch.Tensor:
        """Return synthetic normalized text feature vectors."""
        features = torch.randn(len(text_prompts), self.feature_dim)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def extract_video_features(self, video_path: str, num_frames: int = 8) -> torch.Tensor:
        """Return synthetic video features."""
        features = torch.randn(1, self.feature_dim)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def calculate_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> np.ndarray:
        """Calculate synthetic similarity scores."""
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        if image_features.shape[0] == 1:
            return similarity.cpu().numpy()[0]
        return similarity.cpu().numpy()

class DummyDepthEstimator:
    """Lightweight dummy implementation of DepthProEstimator for testing."""
    
    def __init__(self):
        self.device = "cpu"
    
    def estimate_batch_depth(self, inputs: List) -> List[Dict]:
        """Return synthetic depth estimation results."""
        results = []
        for input_item in inputs:
            if isinstance(input_item, str):
                # If input is a path, create synthetic depth based on filename
                height, width = 480, 640  # Default dimensions
            else:
                # If input is PIL Image
                width, height = input_item.size
            
            # Generate synthetic depth map
            depth_map = np.random.uniform(0.1, 10.0, (height, width)).astype(np.float32)
            focal_length = min(width, height) * 0.7  # Synthetic focal length
            
            result = {
                'depth': depth_map,
                'focallength_px': focal_length
            }
            results.append(result)
        
        return results
```

### 15.6 Test Directory Structure

```
tests/
├── conftest.py                    # Pytest configuration and shared fixtures
├── test_decorators.py             # Test framework decorators (@require_torch_gpu, @slow)
├── test_factories.py              # Dummy model factories and implementations
├── pytest.ini                     # Pytest configuration file
├── lightweight/                   # Fast CPU-only unit tests
│   ├── __init__.py
│   ├── test_config_manager.py     # Configuration loading and validation
│   ├── test_data_models.py        # Pydantic model validation and serialization
│   ├── test_utils.py              # Utility functions (image cropping, polygons)
│   ├── test_parser.py             # String parsing and formatting functions
│   └── test_data_io.py            # File I/O operations (mocked)
├── integration/                   # GPU-heavy integration tests
│   ├── __init__.py
│   ├── test_object_detection.py   # Real GroundingDINO model testing
│   ├── test_object_segmentation.py # Real SAM model testing
│   ├── test_feature_extraction.py # Real CLIP model testing
│   ├── test_depth_estimation.py   # Real DepthPro model testing
│   ├── test_object_listing.py     # Real VLM model testing
│   └── test_processing_pipeline.py # End-to-end pipeline integration
├── mock_based/                    # Tests using dummy models
│   ├── __init__.py
│   ├── test_processing_manager.py # ProcessingManager with dummy models
│   ├──_pipeline_orchestration.py # Pipeline logic testing
│   └── test_error_handling.py     # Error scenarios and edge cases
└── fixtures/                      # Test data and shared fixtures
    ├── __init__.py
    ├── sample_images/             # Test images for various scenarios
    ├── sample_configs/            # Test configuration files
    └── expected_outputs/          # Expected test outputs for validation
```

### 15.7 Example Test Implementations

#### 15.7.1 Lightweight Test Example

```python
# tests/lightweight/test_config_manager.py
import pytest
import tempfile
import os
from pathlib import Path
from cortexia_video.config_manager import ConfigManager

class TestConfigManager:
    """Test suite for ConfigManager - lightweight, CPU-only tests."""
    
    def test_load_valid_toml_config(self):
        """Test loading a valid TOML configuration file."""
        config_content = """
        [object_detection]
        model_name = "IDEA-Research/grounding-dino-base"
        box_threshold = 0.35
        text_threshold = 0.25
        
        [object_segmentation]
        model_name = "facebook/sam-vit-huge"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            
            # Test object detection config
            assert config_manager.get_object_detection_config()['model_name'] == "IDEA-Research/grounding-dino-base"
            assert config_manager.get_object_detection_config()['box_threshold'] == 0.35
            assert config_manager.get_object_detection_config()['text_threshold'] == 0.25
            
            # Test object segmentation config
            assert config_manager.get_object_segmentation_config()['model_name'] == "facebook/sam-vit-huge"
            
        finally:
            os.unlink(config_path)
    
    def test_load_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("invalid toml content [[[")
            config_path = f.name
        
        try:
            with pytest.raises(Exception):  # Should raise parsing error
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager("nonexistent_config.toml")
    
    def test_default_values(self):
        """Test that default values are properly set."""
        config_content = """
        [object_detection]
        model_name = "IDEA-Research/grounding-dino-base"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            od_config = config_manager.get_object_detection_config()
            
            # Test that defaults are applied for missing values
            assert 'box_threshold' in od_config
            assert 'text_threshold' in od_config
            
        finally:
            os.unlink(config_path)
```

#### 15.7.2 Mock-based Test Example

```python
# tests/mock_based/test_processing_manager.py
import pytest
import os
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np
from cortexia_video.processing_manager import ProcessingManager
from cortexia_video.config_manager import ConfigManager
from tests.test_factories import ModelFactory

class TestProcessingManagerWithDummyModels:
    """Test ProcessingManager using dummy models - no GPU required."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test configuration manager."""
        config_content = """
        [object_detection]
        model_name = "IDEA-Research/grounding-dino-base"
        box_threshold = 0.35
        text_threshold = 0.25
        
        [object_segmentation]
        model_name = "facebook/sam-vit-huge"
        
        [feature_extraction]
        model_name = "openbmb/MiniCPM-V-2_6"
        """
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        config_manager = ConfigManager(config_path)
        yield config_manager
        
        os.unlink(config_path)
    
    @pytest.fixture
    def sample_images(self):
        """Create sample test images."""
        images = []
        for i in range(3):
            # Create a simple colored image
            img = Image.new('RGB', (640, 480), color=(i*80, 100, 200))
            images.append(img)
        return images
    
    def test_processing_manager_initialization_with_dummy_models(self, config_manager):
        """Test ProcessingManager initialization using dummy models."""
        # Force use of dummy models
        with patch.dict(os.environ, {'USE_DUMMY_MODELS': 'true'}):
            processing_manager = ProcessingManager(config_manager)
            
            # Verify that dummy models are being used
            assert processing_manager.object_detector.device == "cpu"
            assert processing_manager.object_segmenter.device == "cpu"
            assert processing_manager.feature_extractor.device == "cpu"
    
    def test_object_detection_pipeline_with_dummy_models(self, config_manager, sample_images):
        """Test object detection pipeline using dummy models."""
        with patch.dict(os.environ, {'USE_DUMMY_MODELS': 'true'}):
            processing_manager = ProcessingManager(config_manager)
            
            text_prompts = [["person", "car"], ["dog", "cat"], ["tree"]]
            
            results = processing_manager.object_detector.detect_objects(sample_images, text_prompts)
            
            # Verify structure of dummy results
            assert len(results) == len(sample_images)
            for i, (image_results, prompts) in enumerate(zip(results, text_prompts)):
                assert len(image_results) <= 2  # Dummy model limits to 2 detections
                for detection in image_results:
                    assert 'score' in detection
                    assert 'label' in detection
                    assert 'box' in detection
                    assert detection['label'] in prompts
                    assert 0.0 <= detection['score'] <= 1.0
                    assert len(detection['box']) == 4  # [x1, y1, x2, y2]
    
    def test_segmentation_pipeline_with_dummy_models(self, config_manager, sample_images):
        """Test segmentation pipeline using dummy models."""
        with patch.dict(os.environ, {'USE_DUMMY_MODELS': 'true'}):
            processing_manager = ProcessingManager(config_manager)
            
            # Create synthetic bounding boxes
            batch_input_boxes = [
                [[100, 100, 200, 200], [300, 150, 400, 250]],  # 2 boxes for first image
                [[50, 50, 150, 150]],                           # 1 box for second image
                [[200, 200, 300, 300], [400, 300, 500, 400]]   # 2 boxes for third image
            ]
            
            results = processing_manager.object_segmenter.segment_object(
                sample_images, batch_input_boxes
            )
            
            # Verify structure of dummy segmentation results
            assert len(results) == len(sample_images)
            for i, (image_masks, input_boxes) in enumerate(zip(results, batch_input_boxes)):
                assert len(image_masks) == len(input_boxes)
                for mask, box in zip(image_masks, input_boxes):
                    assert isinstance(mask, np.ndarray)
                    assert mask.dtype == bool
                    assert mask.shape == (sample_images[i].height, sample_images[i].width)
                    
                    # Verify that mask covers the expected region
                    x1, y1, x2, y2 = map(int, box)
                    assert np.any(mask[y1:y2, x1:x2])  # Should have True values in the box region
    
    def test_feature_extraction_pipeline_with_dummy_models(self, config_manager, sample_images):
        """Test feature extraction pipeline using dummy models."""
        with patch.dict(os.environ, {'USE_DUMMY_MODELS': 'true'}):
            processing_manager = ProcessingManager(config_manager)
            
            # Test image feature extraction
            image_features = processing_manager.feature_extractor.extract_image_features(sample_images)
            
            assert image_features.shape == (len(sample_images), 512)  # Standard CLIP dimension
            assert torch.allclose(torch.norm(image_features, dim=-1), torch.ones(len(sample_images)), atol=1e-6)
            
            # Test text feature extraction
            text_prompts = ["a person walking", "a red car", "a beautiful sunset"]
            text_features = processing_manager.feature_extractor.extract_text_features(text_prompts)
            
            assert text_features.shape == (len(text_prompts), 512)
            assert torch.allclose(torch.norm(text_features, dim=-1), torch.ones(len(text_prompts)), atol=1e-6)
            
            # Test similarity calculation
            similarity = processing_manager.feature_extractor.calculate_similarity(image_features, text_features)
            
            assert similarity.shape == (len(sample_images), len(text_prompts))
            assert np.allclose(np.sum(similarity, axis=1), 1.0, atol=1e-6)  # Should sum to 1 (softmax)
    
    def test_end_to_end_pipeline_orchestration_with_dummy_models(self, config_manager, sample_images):
        """Test end-to-end pipeline orchestration using dummy models."""
        with patch.dict(os.environ, {'USE_DUMMY_MODELS': 'true'}):
            processing_manager = ProcessingManager(config_manager)
            
            # Simulate a complete processing pipeline
            text_prompts = [["person", "car"], ["dog"], ["tree", "building"]]
            
            # Step 1: Object detection
            detections = processing_manager.object_detector.detect_objects(sample_images, text_prompts)
            
            # Step 2: Extract bounding boxes for segmentation
            batch_input_boxes = []
            for detection_list in detections:
                boxes = [det['box'] for det in detection_list]
                batch_input_boxes.append(boxes)
            
            # Step 3: Object segmentation
            if any(boxes for boxes in batch_input_boxes):  # Only if we have detections
                segmentation_results = processing_manager.object_segmenter.segment_object(
                    sample_images, batch_input_boxes
                )
                
                # Verify segmentation results match detection results
                for i, (detection_list, masks) in enumerate(zip(detections, segmentation_results)):
                    assert len(masks) == len(detection_list)
            
            # Step 4: Feature extraction
            image_features = processing_manager.feature_extractor.extract_image_features(sample_images)
            
            # Verify complete pipeline results
            assert len(detections) == len(sample_images)
            assert image_features.shape[0] == len(sample_images)
    
    def test_error_handling_with_dummy_models(self, config_manager):
        """Test error handling scenarios using dummy models."""
        with patch.dict(os.environ, {'USE_DUMMY_MODELS': 'true'}):
            processing_manager = ProcessingManager(config_manager)
            
            # Test empty input handling
            empty_results = processing_manager.object_detector.detect_objects([], [])
            assert empty_results == []
            
            empty_features = processing_manager.feature_extractor.extract_image_features([])
            assert empty_features.shape == (0, 512)
            
            # Test mismatched input lengths
            with pytest.raises((ValueError, IndexError)):
                processing_manager.object_detector.detect_objects(
                    sample_images[:2], [["person"], ["car"], ["tree"]]  # Mismatched lengths
                )
```

#### 15.7.3 GPU-heavy Integration Test Example

```python
# tests/integration/test_object_detection.py
import pytest
import torch
from PIL import Image
import numpy as np
from cortexia_video.object_detection import ObjectDetector
from cortexia_video.config_manager import ConfigManager
from tests.test_decorators import require_torch_gpu, slow, require_model_download

class TestObjectDetectionIntegration:
    """Integration tests for ObjectDetector using real GPU models."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test configuration manager."""
        config_content = """
        [object_detection]
        model_name = "IDEA-Research/grounding-dino-base"
        box_threshold = 0.35
        text_threshold = 0.25
        """
        
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        config_manager = ConfigManager(config_path)
        yield config_manager
        
        os.unlink(config_path)
    
    @pytest.fixture
    def sample_images(self):
        """Create sample test images with known objects."""
        # Create more realistic test images
        images = []
        
        # Simple geometric shapes that models can detect
        img1 = Image.new('RGB', (640, 480), color=(255, 255, 255))
        # Add some simple shapes/patterns that might be detectable
        images.append(img1)
        
        return images
    
    @require_torch_gpu
    @require_model_download
    @slow
    def test_real_object_detection_initialization(self, config_manager):
        """Test real ObjectDetector initialization and model loading."""
        detector = ObjectDetector(config_manager)
        
        # Verify GPU usage
        assert detector.device.type == 'cuda'
        assert detector.model is not None
        assert detector.processor is not None
    
    @require_torch_gpu
    @require_model_download
    @slow
    def test_real_object_detection_inference(self, config_manager, sample_images):
        """Test real object detection inference on sample images."""
        detector = ObjectDetector(config_manager)
        
        text_prompts = [["person", "car", "building"]]
        
        results = detector.detect_objects(sample_images[:1], text_prompts)
        
        # Verify result structure
        assert len(results) == 1
        assert isinstance(results[0], list)
        
        # Verify detection format
        for detection in results[0]:
            assert 'score' in detection
            assert 'label' in detection
            assert 'box' in detection
            assert isinstance(detection['score'], (float, np.floating))
            assert isinstance(detection['label'], str)
            assert len(detection['box']) == 4
            assert detection['label'] in text_prompts[0]
    
    @require_torch_gpu
    @require_model_download
    @slow
    def test_real_object_detection_batch_processing(self, config_manager, sample_images):
        """Test real object detection with batch processing
        """Test real object detection with batch processing."""
        detector = ObjectDetector(config_manager)
        
        # Test with multiple images and prompts
        text_prompts = [["person", "car"], ["building", "tree"], ["dog", "cat"]]
        
        results = detector.detect_objects(sample_images, text_prompts)
        
        # Verify batch results
        assert len(results) == len(sample_images)
        assert len(results) == len(text_prompts)
        
        for i, (image_results, prompts) in enumerate(zip(results, text_prompts)):
            assert isinstance(image_results, list)
            for detection in image_results:
                assert detection['label'] in prompts
    
    @require_torch_gpu
    @require_model_download
    @slow
    def test_real_object_detection_memory_usage(self, config_manager, sample_images):
        """Test GPU memory usage during real object detection."""
        initial_memory = torch.cuda.memory_allocated()
        
        detector = ObjectDetector(config_manager)
        model_loaded_memory = torch.cuda.memory_allocated()
        
        # Verify model loading increased memory usage
        assert model_loaded_memory > initial_memory
        
        # Run inference
        text_prompts = [["person", "car"]] * len(sample_images)
        results = detector.detect_objects(sample_images, text_prompts)
        
        inference_memory = torch.cuda.memory_allocated()
        
        # Clean up
        del detector
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        
        # Verify memory was properly released
        assert final_memory <= initial_memory * 1.1  # Allow small tolerance
```

### 15.8 Running Different Test Suites

#### 15.8.1 Development Workflow Commands

```bash
# Quick development tests (lightweight only)
pytest tests/lightweight/ -v

# Mock-based tests for integration logic
pytest tests/mock_based/ -v

# Lightweight + mock-based (recommended for development)
pytest tests/lightweight/ tests/mock_based/ -v

# Skip slow tests explicitly
pytest tests/ -m "not slow" -v

# Skip GPU tests explicitly
SKIP_GPU=true pytest tests/ -v
```

#### 15.8.2 CI/CD Integration Commands

```bash
# Pull Request CI (fast, no GPU)
SKIP_GPU=true pytest tests/lightweight/ tests/mock_based/ -v --cov=cortexia_video

# Main Branch CI (full test suite)
RUN_SLOW=true pytest tests/ -v --cov=cortexia_video --cov-report=html

# GPU-only tests
RUN_SLOW=true pytest tests/integration/ -m gpu -v

# Performance tests
RUN_SLOW=true pytest tests/performance/ -v --benchmark-only
```

#### 15.8.3 Pytest Configuration

```ini
# pytest.ini
[tool:pytest]
markers =
    gpu: marks tests as requiring GPU (deselect with '-m "not gpu"')
    slow: marks tests as slow (deselect with '-m "not slow"')
    model_download: marks tests requiring model downloads
    integration: marks integration tests
    unit: marks unit tests

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Ignore warnings from third-party libraries
filterwarnings =
    ignore::UserWarning:transformers.*
    ignore::FutureWarning:torch.*
    ignore::DeprecationWarning:numpy.*

# Test discovery patterns
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --durations=10
```

### 15.9 CI/CD Integration Guidelines

#### 15.9.1 GitHub Actions Workflow Example

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  lightweight-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run lightweight tests
      env:
        SKIP_GPU: true
        USE_DUMMY_MODELS: true
      run: |
        pytest tests/lightweight/ tests/mock_based/ -v --cov=cortexia_video --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  gpu-integration-tests:
    runs-on: [self-hosted, gpu]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run GPU integration tests
      env:
        RUN_SLOW: true
        CUDA_VISIBLE_DEVICES: 0
      run: |
        pytest tests/integration/ -v -m gpu --timeout=300
    
    - name: Run performance tests
      env:
        RUN_SLOW: true
        CUDA_VISIBLE_DEVICES: 0
      run: |
        pytest tests/performance/ -v --benchmark-only
```

#### 15.9.2 Docker-based Testing

```dockerfile
# Dockerfile.test
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install pytest pytest-cov pytest-benchmark

COPY . .

# Default command runs lightweight tests
CMD ["pytest", "tests/lightweight/", "tests/mock_based/", "-v"]
```

```bash
# Build and run test container
docker build -f Dockerfile.test -t cortexia-video-test .

# Run lightweight tests
docker run --rm cortexia-video-test

# Run GPU tests (requires nvidia-docker)
docker run --rm --gpus all -e RUN_SLOW=true cortexia-video-test \
    pytest tests/integration/ -v -m gpu
```

### 15.10 Test Configuration Management

#### 15.10.1 Shared Test Fixtures

```python
# tests/conftest.py
import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from cortexia_video.config_manager import ConfigManager

@pytest.fixture(scope="session")
def test_config_manager():
    """Create a shared test configuration manager."""
    config_content = """
    [model_settings]
    object_listing_model = "recognize_anything/ram"
    object_detection_model = "IDEA-Research/grounding-dino-base"
    segmentation_model = "facebook/sam-vit-base"
    description_model = "nvidia/DAM-3B-Self-Contained"
    clip_feature_model = "PE-Core-B16-224"
    clip_feature_model_identifier = "clip_pe"
    
    [detection_settings]
    box_threshold = 0.35
    text_threshold = 0.25
    
    [processing]
    default_mode = "list | detect | segment | describe | extract_features"
    frame_interval = 30
    batch_size = 2
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    config_manager = ConfigManager(config_file_path=config_path)
    config_manager.load_config()
    
    yield config_manager
    
    os.unlink(config_path)

@pytest.fixture
def sample_images():
    """Create sample test images for various test scenarios."""
    images = []
    
    # Create images with different characteristics
    for i in range(3):
        # Create a simple colored image with some patterns
        img = Image.new('RGB', (640, 480), color=(i*80, 100, 200))
        
        # Add some simple geometric shapes for detection
        pixels = np.array(img)
        
        # Add a rectangle
        pixels[100:200, 100:200] = [255, 0, 0]  # Red rectangle
        
        # Add a circle (approximate)
        center_x, center_y = 400, 300
        radius = 50
        y, x = np.ogrid[:480, :640]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        pixels[mask] = [0, 255, 0]  # Green circle
        
        img = Image.fromarray(pixels)
        images.append(img)
    
    return images

@pytest.fixture
def sample_video_path(tmp_path):
    """Create a sample video file for testing."""
    # This would create a simple test video
    # For now, return a placeholder path
    video_path = tmp_path / "test_video.mp4"
    # In a real implementation, you'd create a simple test video here
    return str(video_path)

@pytest.fixture
def mock_detection_results():
    """Provide mock detection results for testing."""
    return [
        [
            {'score': 0.95, 'label': 'person', 'box': [100, 100, 200, 300]},
            {'score': 0.87, 'label': 'car', 'box': [300, 150, 500, 250]}
        ],
        [
            {'score': 0.92, 'label': 'dog', 'box': [50, 80, 150, 200]}
        ],
        []  # Empty results for third image
    ]

@pytest.fixture
def mock_segmentation_masks():
    """Provide mock segmentation masks for testing."""
    masks = []
    for i in range(3):
        image_masks = []
        if i < 2:  # First two images have masks
            mask = np.zeros((480, 640), dtype=bool)
            mask[100:200, 100:200] = True  # Simple rectangular mask
            image_masks.append(mask)
        masks.append(image_masks)
    return masks

@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically clean up GPU memory after each test."""
    yield
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
```

#### 15.10.2 Test Data Management

```python
# tests/fixtures/test_data_manager.py
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import requests
from PIL import Image

class TestDataManager:
    """Manages test data downloads and caching."""
    
    def __init__(self, cache_dir: str = "tests/fixtures/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_sample_image(self, image_name: str) -> Image.Image:
        """Get a sample image, downloading if necessary."""
        image_path = self.cache_dir / f"{image_name}.jpg"
        
        if not image_path.exists():
            self._download_sample_image(image_name, image_path)
        
        return Image.open(image_path)
    
    def get_expected_results(self, test_name: str) -> Dict[str, Any]:
        """Get expected test results from cached JSON files."""
        results_path = self.cache_dir / f"{test_name}_expected.json"
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        
        return {}
    
    def save_expected_results(self, test_name: str, results: Dict[str, Any]):
        """Save expected test results for future comparisons."""
        results_path = self.cache_dir / f"{test_name}_expected.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _download_sample_image(self, image_name: str, save_path: Path):
        """Download a sample image from a test dataset."""
        # This would download from a test image repository
        # For now, create a simple placeholder image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(save_path)

# Global test data manager instance
test_data_manager = TestDataManager()
```

### 15.11 Performance and Memory Testing

#### 15.11.1 Memory Usage Tests

```python
# tests/performance/test_memory_usage.py
import pytest
import torch
import psutil
import gc
from cortexia_video.processing_manager import ProcessingManager
from tests.test_decorators import require_torch_gpu, slow

class TestMemoryUsage:
    """Test memory usage and leak detection."""
    
    @require_torch_gpu
    @slow
    def test_gpu_memory_stability(self, test_config_manager, sample_images):
        """Test that GPU memory usage remains stable across multiple processing cycles."""
        initial_memory = torch.cuda.memory_allocated()
        
        for i in range(10):
            processing_manager = ProcessingManager(test_config_manager)
            
            # Process images
            text_prompts = [["person", "car"]] * len(sample_images)
            results = processing_manager.object_detector.detect_objects(sample_images, text_prompts)
            
            # Clean up
            del processing_manager
            torch.cuda.empty_cache()
            gc.collect()
            
            current_memory = torch.cuda.memory_allocated()
            
            # Memory should not grow significantly
            assert current_memory <= initial_memory * 1.1, f"Memory leak detected at iteration {i}"
    
    def test_cpu_memory_stability(self, test_config_manager, sample_images):
        """Test CPU memory stability using dummy models."""
        import os
        with patch.dict(os.environ, {'USE_DUMMY_MODELS': 'true'}):
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            for i in range(50):
                processing_manager = ProcessingManager(test_config_manager)
                
                # Process images with dummy models
                text_prompts = [["person", "car"]] * len(sample_images)
                results = processing_manager.object_detector.detect_objects(sample_images, text_prompts)
                
                del processing_manager
                gc.collect()
                
                current_memory = process.memory_info().rss
                memory_growth = (current_memory - initial_memory) / initial_memory
                
                # Allow up to 10% memory growth
                assert memory_growth <= 0.1, f"CPU memory leak detected at iteration {i}"
    
    @require_torch_gpu
    @slow
    def test_batch_processing_memory_scaling(self, test_config_manager):
        """Test memory usage scales appropriately with batch size."""
        from cortexia_video.object_detector import ObjectDetector
        
        detector = ObjectDetector(test_config_manager)
        base_memory = torch.cuda.memory_allocated()
        
        batch_sizes = [1, 2, 4, 8]
        memory_usage = []
        
        for batch_size in batch_sizes:
            # Create batch of images
            images = [Image.new('RGB', (640, 480), color=(128, 128, 128)) for _ in range(batch_size)]
            text_prompts = [["person", "car"]] * batch_size
            
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
            results = detector.detect_objects(images, text_prompts)
            
            memory_after = torch.cuda.memory_allocated()
            memory_used = memory_after - memory_before
            memory_usage.append(memory_used)
        
        # Memory usage should scale roughly linearly with batch size
        for i in range(1, len(batch_sizes)):
            ratio = memory_usage[i] / memory_usage[0]
            expected_ratio = batch_sizes[i] / batch_sizes[0]
            
            # Allow 50% tolerance for memory scaling
            assert 0.5 * expected_ratio <= ratio <= 2.0 * expected_ratio
```

#### 15.11.2 Performance Benchmarking

```python
# tests/performance/test_benchmarks.py
import pytest
import time
from cortexia_video.processing_manager import ProcessingManager
from tests.test_decorators import require_torch_gpu, slow

class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @require_torch_gpu
    @slow
    @pytest.mark.benchmark
    def test_object_detection_throughput(self, test_config_manager, sample_images, benchmark):
        """Benchmark object detection throughput."""
        processing_manager = ProcessingManager(test_config_manager)
        text_prompts = [["person", "car", "dog"]] * len(sample_images)
        
        def run_detection():
            return processing_manager.object_detector.detect_objects(sample_images, text_prompts)
        
        result = benchmark(run_detection)
        
        # Verify results are valid
        assert len(result) == len(sample_images)
        
        # Performance assertions (adjust based on your hardware)
        assert benchmark.stats.mean < 5.0  # Should complete in under 5 seconds on average
    
    @slow
    @pytest.mark.benchmark
    def test_dummy_model_performance(self, test_config_manager, sample_images, benchmark):
        """Benchmark dummy model performance for development testing."""
        import os
        with patch.dict(os.environ, {'USE_DUMMY_MODELS': 'true'}):
            processing_manager = ProcessingManager(test_config_manager)
            text_prompts = [["person", "car", "dog"]] * len(sample_images)
            
            def run_detection():
                return processing_manager.object_detector.detect_objects(sample_images, text_prompts)
            
            result = benchmark(run_detection)
            
            # Dummy models should be very fast
            assert benchmark.stats.mean < 0.1  # Should complete in under 100ms
    
    @require_torch_gpu
    @slow
    def test_processing_pipeline_end_to_end(self, test_config_manager, sample_images):
        """Test end-to-end processing pipeline performance."""
        processing_manager = ProcessingManager(test_config_manager)
        
        start_time = time.time()
        
        # Run full pipeline
        results = processing_manager.process_images(
            sample_images,
            modes=["list", "detect", "segment"],
            text_prompts=[["person", "car"]] * len(sample_images)
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        assert len(results) == len(sample_images)
        
        # Performance check (adjust based on your requirements)
        images_per_second = len(sample_images) / processing_time
        assert images_per_second > 0.1  # At least 0.1 images per second
        
        print(f"Processing throughput: {images_per_second:.2f} images/second")
```

This comprehensive test framework provides:

1. **Multi-tier testing strategy** with lightweight, mock-based, and integration tests
2. **Smart test categorization** using decorators for GPU, slow, and model download requirements
3. **Dummy model factories** for fast development testing
4. **Comprehensive CI/CD integration** with different test suites for different scenarios
5. **Memory and performance testing** to catch regressions
6. **Flexible configuration management** for different testing environments
7. **Test data management** with caching and expected results comparison

The framework allows developers to run fast tests during development while ensuring comprehensive testing in CI/CD pipelines, making it practical for both daily development and production deployment.

## 16. Documentation and Knowledge Management

### 16.1 Current Documentation Assessment

The project currently has basic documentation but lacks comprehensive coverage of advanced features and integration patterns. Key gaps include:

- Limited API documentation with usage examples
- Missing architectural decision records (ADRs)
- Insufficient troubleshooting guides
- Lack of performance optimization documentation
- Missing integration examples for different use cases

### 16.2 Comprehensive Documentation Strategy

#### 16.2.1 Documentation Structure Redesign

```
docs/
├── README.md                    # Project overview and quick start
├── INSTALLATION.md             # Detailed installation guide
├── API_REFERENCE.md            # Complete API documentation
├── USER_GUIDE.md              # Comprehensive user guide
├── DEVELOPER_GUIDE.md         # Development setup and contribution
├── ARCHITECTURE.md            # System architecture and design
├── PERFORMANCE.md             # Performance optimization guide
├── TROUBLESHOOTING.md         # Common issues and solutions
├── CHANGELOG.md               # Version history and changes
├── examples/                  # Code examples and tutorials
│   ├── basic_usage/
│   ├── advanced_features/
│   ├── integration_patterns/
│   └── performance_optimization/
├── architecture/              # Architecture documentation
│   ├── decisions/            # Architecture Decision Records
│   ├── diagrams/            # System diagrams
│   └── design_patterns/     # Design pattern documentation
└── deployment/              # Deployment guides
    ├── docker/
    ├── cloud/
    └── production/
```

#### 16.2.2 API Documentation Enhancement

```python
# Enhanced API documentation example
class ObjectDetector:
    """
    Advanced object detection using CLIP and Grounding DINO models.
    
    This class provides state-of-the-art object detection capabilities with support
    for natural language queries, batch processing, and GPU acceleration.
    
    Attributes:
        config_manager (ConfigManager): Configuration management instance
        model: Loaded Grounding DINO model for object detection
        processor: Model processor for input/output handling
        device (str): Computing device ('cuda' or 'cpu')
    
    Example:
        Basic object detection:
        
        >>> from cortexia_video.config_manager import ConfigManager
        >>> from cortexia_video.object_detector import ObjectDetector
        >>> from PIL import Image
        >>> 
        >>> config = ConfigManager()
        >>> config.load_config()
        >>> detector = ObjectDetector(config)
        >>> 
        >>> image = Image.open("sample.jpg")
        >>> results = detector.detect_objects([image], [["person", "car"]])
        >>> print(f"Found {len(results[0])} objects")
        
        Batch processing with different prompts:
        
        >>> images = [Image.open(f"image_{i}.jpg") for i in range(3)]
        >>> prompts = [["person"], ["car", "truck"], ["dog", "cat"]]
        >>> results = detector.detect_objects(images, prompts)
        >>> 
        >>> for i, (image_results, prompt) in enumerate(zip(results, prompts)):
        >>>     print(f"Image {i} with prompts {prompt}: {len(image_results)} detections")
    
    Performance Notes:
        - First inference may be slower due to model loading
        - Batch processing is more efficient than individual image processing
        - GPU acceleration provides 5-10x speedup over CPU
        - Memory usage scales linearly with batch size
    
    See Also:
        - :class:`ProcessingManager`: For complete processing pipelines
        - :class:`ConfigManager`: For configuration management
        - :doc:`examples/object_detection`: Detailed usage examples
    """
    
    def detect_objects(
        self, 
        images: List[Image.Image], 
        text_prompts: List[List[str]],
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in images using natural language prompts.
        
        This method performs object detection on a batch of images using
        corresponding text prompts. Each image can have different prompts,
        allowing for flexible and targeted object detection.
        
        Args:
            images (List[Image.Image]): List of PIL Images to process.
                Images should be in RGB format. Recommended size: 640x640 pixels
                for optimal performance, but any size is supported.
                
            text_prompts (List[List[str]]): List of text prompt lists, one per image.
                Each inner list contains object names to detect in the corresponding
                image. Examples: [["person", "car"], ["dog"], ["building", "tree"]]
                
            confidence_threshold (Optional[float], optional): Minimum confidence
                score for detections. Range: 0.0-1.0. If None, uses config value.
                Higher values = fewer, more confident detections.
                
            nms_threshold (Optional[float], optional): Non-maximum suppression
                threshold for removing overlapping detections. Range: 0.0-1.0.
                Lower values = more aggressive overlap removal.
                
            max_detections (Optional[int], optional): Maximum number of detections
                per image. If None, returns all detections above threshold.
        
        Returns:
            List[List[Dict[str, Any]]]: Nested list structure where:
                - Outer list: One entry per input image
                - Inner list: Detections for that image
                - Dict entries contain:
                    - 'label' (str): Detected object label
                    - 'score' (float): Confidence score (0.0-1.0)
                    - 'box' (List[int]): Bounding box [x1, y1, x2, y2]
                    - 'area' (int): Bounding box area in pixels
                    - 'center' (Tuple[int, int]): Box center coordinates
        
        Raises:
            ValueError: If images and text_prompts have different lengths
            RuntimeError: If model inference fails
            MemoryError: If batch size exceeds available GPU memory
            
        Example:
            Single image detection:
            
            >>> image = Image.open("street_scene.jpg")
            >>> results = detector.detect_objects([image], [["person", "car", "bicycle"]])
            >>> 
            >>> for detection in results[0]:
            >>>     print(f"Found {detection['label']} with confidence {detection['score']:.2f}")
            >>>     x1, y1, x2, y2 = detection['box']
            >>>     print(f"Location: ({x1}, {y1}) to ({x2}, {y2})")
            
            Batch processing with confidence filtering:
            
            >>> images = load_image_batch("dataset/")
            >>> prompts = [["person"]] * len(images)
            >>> results = detector.detect_objects(
            >>>     images, prompts, 
            >>>     confidence_threshold=0.8,
            >>>     max_detections=10
            >>> )
            >>> 
            >>> high_confidence_count = sum(
            >>>     len([d for d in img_results if d['score'] > 0.9])
            >>>     for img_results in results
            >>> )
        
        Performance Tips:
            - Use batch processing for multiple images when possible
            - Resize very large images (>2000px) for faster processing
            - Use specific prompts rather than generic ones for better accuracy
            - Consider confidence_threshold tuning based on your use case
            
        Memory Usage:
            - Approximate GPU memory per image: 200-500MB (depends on resolution)
            - Batch size recommendations:
                - 8GB GPU: 4-8 images per batch
                - 16GB GPU: 8-16 images per batch
                - 24GB+ GPU: 16+ images per batch
        """
        # Implementation details...
```

### 16.3 Interactive Documentation System

#### 16.3.1 Jupyter Notebook Tutorials

```python
# docs/examples/notebooks/01_getting_started.ipynb
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CortexIA Video Processing - Getting Started\n",
    "\n",
    "This notebook provides a hands-on introduction to CortexIA Video Processing capabilities.\n",
    "\n",
    "## What you'll learn:\n",
    "- Basic setup and configuration\n",
    "- Object detection with natural language\n",
    "- Image segmentation\n",
    "- Feature extraction\n",
    "- Performance optimization tips\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Installation and setup\n",
    "!pip install cortexia-video\n",
    "\n",
    "# Import required libraries\n",
    "import torch\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "from cortexia_video.config_manager import ConfigManager\n",
    "from cortexia_video.processing_manager import ProcessingManager\n",
    "\n",
    "# Check GPU availability\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuration setup\n",
    "config_manager = ConfigManager()\n",
    "config_manager.load_config()\n",
    "\n",
    "# Display current configuration\n",
    "print(\"Current Configuration:\")\n",
    "print(f\"Object Detection Model: {config_manager.get_model_setting('object_detection_model')}\")\n",
    "print(f\"Segmentation Model: {config_manager.get_model_setting('segmentation_model')}\")\n",
    "print(f\"Box Threshold: {config_manager.get_detection_setting('box_threshold')}\")\n",
    "print(f\"Text Threshold: {config_manager.get_detection_setting('text_threshold')}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load sample images\n",
    "def load_sample_images():\n",
    "    \"\"\"Load sample images for demonstration.\"\"\"\n",
    "    # You can replace these with your own images\n",
    "    sample_urls = [\n",
    