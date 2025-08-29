"""Tests for base feature classes and functionality"""

import pytest
import numpy as np
import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Type, Union

from cortexia.features.base import BaseFeature
from cortexia.data.models.video import VideoFramePacket
from cortexia.data.models.result.base_result import BaseResult
from cortexia.data.models.field_validation import FrameField
from cortexia.api.exceptions import ProcessingError


class MockResult(BaseResult):
    """Mock result class for testing"""
    
    def __init__(self, test_field: str, confidence: float = 0.5):
        super().__init__(test_field=test_field, confidence=confidence)
    
    def _get_repr_fields(self) -> str:
        return f"test_field={self.test_field}, confidence={self.confidence}"


class MockFeature(BaseFeature):
    """Mock feature implementation for testing"""
    
    output_schema = MockResult
    required_inputs = []
    required_fields = []
    
    def __init__(self, config=None):
        self._process_frame_called = False
        self._initialize_called = False
        super().__init__(config)
    
    def _initialize(self):
        self._initialize_called = True
        self.initialized = True
    
    def process_frame(self, frame: VideoFramePacket, **inputs) -> BaseResult:
        self._process_frame_called = True
        return MockResult(test_field="test_result", confidence=0.8)
    
    @property
    def name(self) -> str:
        return "mock_feature"
    
    @property
    def description(self) -> str:
        return "Mock feature for testing"


class TestBaseFeature:
    """Test cases for BaseFeature class"""
    
    def test_init_with_config(self):
        """Test initialization with configuration"""
        config = {"param1": "value1", "nested": {"param2": "value2"}}
        feature = MockFeature(config)
        
        assert feature.config == config
        assert feature.initialized is True
        assert feature._initialize_called is True
    
    def test_init_without_config(self):
        """Test initialization without configuration"""
        feature = MockFeature()
        
        assert feature.config == {}
        assert feature.initialized is True
        assert feature._initialize_called is True
    
    def test_output_schema_validation(self):
        """Test that output_schema is properly validated"""
        
        class InvalidFeature(BaseFeature):
            def _initialize(self):
                pass
            
            def process_frame(self, frame, **inputs):
                pass
            
            @property
            def name(self):
                return "invalid"
            
            @property
            def description(self):
                return "Invalid feature"
        
        with pytest.raises(ValueError, match="must define output_schema"):
            InvalidFeature()
    
    def test_get_output_schema(self):
        """Test get_output_schema method"""
        feature = MockFeature()
        assert feature.get_output_schema() == MockResult
    
    def test_get_required_fields(self):
        """Test get_required_fields method"""
        feature = MockFeature()
        assert feature.get_required_fields() == []
        
        # Modify class attribute for testing
        MockFeature.required_fields = ["rgb", FrameField.FRAME_DATA]
        assert feature.get_required_fields() == ["rgb", FrameField.FRAME_DATA]
        
        # Reset for other tests
        MockFeature.required_fields = []
    
    def test_validate_frame_inputs_no_requirements(self):
        """Test frame validation when no fields are required"""
        feature = MockFeature()
        frame = self._create_test_frame()
        
        # Should not raise any exception
        feature.validate_frame_inputs(frame)
    
    def test_validate_frame_inputs_with_requirements(self):
        """Test frame validation when fields are required"""
        feature = MockFeature()
        # Modify class attribute for testing
        MockFeature.required_fields = [FrameField.FRAME_DATA]
        frame = self._create_test_frame()
        
        # Should not raise any exception
        feature.validate_frame_inputs(frame)
        
        # Reset for other tests
        MockFeature.required_fields = []
    
    def test_validate_inputs_no_requirements(self):
        """Test inputs validation when no inputs are required"""
        feature = MockFeature()
        
        # Should not raise any exception
        feature.validate_inputs()
    
    def test_validate_inputs_with_requirements(self):
        """Test inputs validation when inputs are required"""
        feature = MockFeature()
        feature.required_inputs = ["input1", "input2"]
        
        # Should raise exception when required inputs are missing
        with pytest.raises(ProcessingError, match="missing required inputs"):
            feature.validate_inputs()
        
        # Should not raise exception when all required inputs are provided
        feature.validate_inputs(input1="value1", input2="value2")
    
    def test_process_frame(self):
        """Test process_frame method"""
        feature = MockFeature()
        frame = self._create_test_frame()
        
        result = feature.process_frame(frame)
        
        assert isinstance(result, MockResult)
        assert result.test_field == "test_result"
        assert result.confidence == 0.8
        assert feature._process_frame_called is True
    
    def test_process_batch(self):
        """Test process_batch method"""
        feature = MockFeature()
        frames = [self._create_test_frame(i) for i in range(3)]
        
        results = feature.process_batch(frames)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, MockResult)
            assert result.test_field == "test_result"
            assert result.confidence == 0.8
    
    def test_process_batch_with_inputs_validation(self):
        """Test process_batch with input validation"""
        feature = MockFeature()
        feature.required_inputs = ["required_input"]
        frames = [self._create_test_frame()]
        
        # Should raise exception when required inputs are missing
        with pytest.raises(ProcessingError, match="missing required inputs"):
            feature.process_batch(frames)
        
        # Should work when required inputs are provided
        results = feature.process_batch(frames, required_input="value")
        assert len(results) == 1
    
    def test_name_property(self):
        """Test name property"""
        feature = MockFeature()
        assert feature.name == "mock_feature"
    
    def test_description_property(self):
        """Test description property"""
        feature = MockFeature()
        assert feature.description == "Mock feature for testing"
    
    def test_validate_config(self):
        """Test validate_config method"""
        feature = MockFeature()
        assert feature.validate_config() is True
    
    def test_get_config_param(self):
        """Test get_config_param method"""
        config = {
            "param1": "value1",
            "nested": {
                "param2": "value2",
                "deep": {"param3": "value3"}
            }
        }
        feature = MockFeature(config)
        
        assert feature.get_config_param("param1") == "value1"
        assert feature.get_config_param("nested.param2") == "value2"
        assert feature.get_config_param("nested.deep.param3") == "value3"
        assert feature.get_config_param("nonexistent") is None
        assert feature.get_config_param("nonexistent", "default") == "default"
    
    def test_set_config_param(self):
        """Test set_config_param method"""
        feature = MockFeature()
        
        feature.set_config_param("param1", "value1")
        assert feature.config["param1"] == "value1"
        
        feature.set_config_param("nested.param2", "value2")
        assert feature.config["nested"]["param2"] == "value2"
        
        feature.set_config_param("existing.new", "value3")
        assert feature.config["existing"]["new"] == "value3"
    
    def test_is_ready(self):
        """Test is_ready method"""
        feature = MockFeature()
        assert feature.is_ready() is True
        
        feature.initialized = False
        assert feature.is_ready() is False
    
    def test_repr(self):
        """Test __repr__ method"""
        feature = MockFeature()
        repr_str = repr(feature)
        assert "MockFeature" in repr_str
        assert "name='mock_feature'" in repr_str
        assert "initialized=True" in repr_str
    
    def _create_test_frame(self, frame_number=0):
        """Helper method to create test VideoFramePacket"""
        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        timestamp = datetime.timedelta(seconds=frame_number / 30.0)
        
        # Create minimal trajectory data
        from cortexia.data.models.video import TrajectoryPoint
        trajectory_point = TrajectoryPoint(
            x=frame_number * 0.1,
            y=frame_number * 0.1,
            z=0.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0
        )
        
        return VideoFramePacket(
            frame_data=frame_data,
            frame_number=frame_number,
            timestamp=timestamp,
            source_video_id="test_video",
            trajecotry=[trajectory_point],
            current_traj_index=0
        )


class TestBaseFeatureIntegration:
    """Integration tests for BaseFeature with real data"""
    
    def test_process_frame_with_real_image(self):
        """Test processing frame with real image data"""
        feature = MockFeature()
        
        # Create a realistic image frame
        frame_data = np.zeros((224, 224, 3), dtype=np.uint8)
        frame_data[:, :, 0] = 255  # Red channel
        
        frame = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video"
        )
        
        result = feature.process_frame(frame)
        
        assert isinstance(result, MockResult)
        assert result.test_field == "test_result"
    
    def test_process_batch_with_mixed_frames(self):
        """Test processing batch with mixed valid and invalid frames"""
        feature = MockFeature()
        
        valid_frame = self._create_test_frame(0)
        # Create invalid frame with empty data instead of None
        invalid_frame = VideoFramePacket(
            frame_data=np.zeros((1, 1, 3), dtype=np.uint8),
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1/30.0),
            source_video_id="test_video"
        )
        
        frames = [valid_frame, invalid_frame, valid_frame]
        results = feature.process_batch(frames)
        
        assert len(results) == 3
        # All results should be returned, even for invalid frames
        for result in results:
            assert isinstance(result, MockResult)
    
    def _create_test_frame(self, frame_number=0):
        """Helper method to create test VideoFramePacket"""
        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        timestamp = datetime.timedelta(seconds=frame_number / 30.0)
        
        # Create minimal trajectory data
        from cortexia.data.models.video import TrajectoryPoint
        trajectory_point = TrajectoryPoint(
            x=frame_number * 0.1,
            y=frame_number * 0.1,
            z=0.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0
        )
        
        return VideoFramePacket(
            frame_data=frame_data,
            frame_number=frame_number,
            timestamp=timestamp,
            source_video_id="test_video",
            trajecotry=[trajectory_point],
            current_traj_index=0
        )