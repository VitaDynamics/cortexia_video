"""Tests for all gate implementations"""

import pytest
import numpy as np
import datetime
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from cortexia.gates.base_gate import BaseGate
from cortexia.gates.blur_gate import BlurGate
from cortexia.gates.clip_gate import ClipGate
from cortexia.gates.entropy_gate import EntropyGate
from cortexia.gates.grid_gate import GridGate
from cortexia.gates.hash_gate import HashGate
from cortexia.data.models.video import VideoFramePacket
from cortexia.data.models.result.gate_result import GateResult
from cortexia.api.exceptions import ProcessingError


class TestBaseGate:
    """Test cases for BaseGate class"""
    
    def test_base_gate_abstract_method(self):
        """Test that BaseGate cannot be instantiated directly"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseGate()
    
    def test_base_gate_requires_output_schema(self):
        """Test that subclasses must define output_schema"""
        
        class InvalidGate(BaseGate):
            def process_frame(self, frame, **inputs):
                pass
            
            @property
            def name(self):
                return "invalid"
        
        with pytest.raises(ValueError, match="must define output_schema"):
            InvalidGate()


class TestBlurGate:
    """Test cases for BlurGate"""
    
    def test_init(self):
        """Test initialization"""
        gate = BlurGate()
        assert gate.session_id == "default_session"
        assert hasattr(gate, 'logger')
    
    def test_calculate_blur_score_empty_image(self):
        """Test blur score calculation with empty image"""
        gate = BlurGate()
        empty_image = np.array([], dtype=np.uint8)
        score = gate._calculate_blur_score(empty_image)
        assert score == 0.0
    
    def test_calculate_blur_score_zero_size_image(self):
        """Test blur score calculation with zero size image"""
        gate = BlurGate()
        zero_size_image = np.zeros((0, 0, 3), dtype=np.uint8)
        score = gate._calculate_blur_score(zero_size_image)
        assert score == 0.0
    
    def test_calculate_blur_score_solid_color(self):
        """Test blur score calculation with solid color image"""
        gate = BlurGate()
        solid_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        score = gate._calculate_blur_score(solid_image)
        # Solid color should have low blur score (low variance)
        assert score < 10.0
    
    def test_calculate_blur_score_grayscale(self):
        """Test blur score calculation with grayscale image"""
        gate = BlurGate()
        grayscale_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        score = gate._calculate_blur_score(grayscale_image)
        assert isinstance(score, float)
    
    def test_process_frame(self):
        """Test process_frame method"""
        gate = BlurGate()
        frame = self._create_test_frame()
        
        result = gate.process_frame(frame)
        
        assert isinstance(result, GateResult)
        assert result.gate_name == "blur_gate"
        assert isinstance(result.score, float)
        assert result.score >= 0.0
        assert result.metadata is not None
        assert "metric" in result.metadata
        assert result.metadata["metric"] == "variance_of_laplacian"
    
    def _create_test_frame(self, frame_number=0):
        """Helper method to create test VideoFramePacket"""
        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        timestamp = datetime.timedelta(seconds=frame_number / 30.0)
        
        return VideoFramePacket(
            frame_data=frame_data,
            frame_number=frame_number,
            timestamp=timestamp,
            source_video_id="test_video"
        )


class TestEntropyGate:
    """Test cases for EntropyGate"""
    
    def test_init_default_threshold(self):
        """Test initialization with default threshold"""
        gate = EntropyGate()
        assert gate.threshold == 4.0
        assert gate.session_id == "default_session"
    
    def test_init_custom_threshold(self):
        """Test initialization with custom threshold"""
        gate = EntropyGate(threshold=5.5)
        assert gate.threshold == 5.5
    
    def test_calculate_entropy_empty_image(self):
        """Test entropy calculation with empty image"""
        gate = EntropyGate()
        empty_image = np.array([], dtype=np.uint8)
        entropy = gate._calculate_entropy(empty_image)
        assert entropy == 0.0
    
    def test_calculate_entropy_solid_color(self):
        """Test entropy calculation with solid color image"""
        gate = EntropyGate()
        solid_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        entropy = gate._calculate_entropy(solid_image)
        # Solid color should have low entropy
        assert entropy < 1.0
    
    def test_calculate_entropy_grayscale(self):
        """Test entropy calculation with grayscale image"""
        gate = EntropyGate()
        grayscale_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        entropy = gate._calculate_entropy(grayscale_image)
        assert isinstance(entropy, float)
        assert entropy >= 0.0
    
    def test_process_frame_high_entropy(self):
        """Test process_frame with high entropy image"""
        gate = EntropyGate(threshold=2.0)
        frame = self._create_test_frame()
        
        result = gate.process_frame(frame)
        
        assert isinstance(result, GateResult)
        assert result.gate_name == "entropy_gate"
        assert isinstance(result.score, float)
        assert result.score >= 0.0
        assert result.threshold == 2.0
        assert result.metadata is not None
        assert "is_low_entropy" in result.metadata
        assert "entropy_score" in result.metadata
    
    def test_process_frame_low_entropy(self):
        """Test process_frame with low entropy image"""
        gate = EntropyGate(threshold=8.0)  # High threshold
        frame = self._create_test_frame()
        
        result = gate.process_frame(frame)
        
        assert isinstance(result, GateResult)
        assert result.score is not None
        assert result.score < 8.0  # Should be below threshold
        assert result.metadata is not None
        assert result.metadata["is_low_entropy"] is True
    
    def _create_test_frame(self, frame_number=0):
        """Helper method to create test VideoFramePacket"""
        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        timestamp = datetime.timedelta(seconds=frame_number / 30.0)
        
        return VideoFramePacket(
            frame_data=frame_data,
            frame_number=frame_number,
            timestamp=timestamp,
            source_video_id="test_video"
        )