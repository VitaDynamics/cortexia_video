"""Tests for caption feature"""

import pytest
import numpy as np
import datetime
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from cortexia.features.caption.captioner import CaptionFeature
from cortexia.data.models.video import VideoFramePacket
from cortexia.data.models.result.caption_result import CaptionResult
from cortexia.api.exceptions import ModelLoadError, ProcessingError


class TestCaptionFeature:
    """Test cases for CaptionFeature"""
    
    def test_init_with_config(self):
        """Test initialization with configuration"""
        config = {
            "model": "test/model",
            "revision": "test-revision",
            "caption_length": "short"
        }
        
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.return_value = Mock()
            feature = CaptionFeature(config)
            
            assert feature.config == config
            assert feature.device is not None
    
    def test_init_without_config(self):
        """Test initialization without configuration"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.return_value = Mock()
            feature = CaptionFeature()
            
            assert feature.config == {}
    
    def test_initialize_success(self):
        """Test successful initialization"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.return_value = Mock()
            feature = CaptionFeature()
            
            assert feature.initialized is True
            assert feature.captioner is not None
    
    def test_initialize_failure(self):
        """Test initialization failure"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.side_effect = Exception("Model load failed")
            
            with pytest.raises(ModelLoadError, match="Failed to initialize captioning model"):
                CaptionFeature()
    
    def test_name_property(self):
        """Test name property"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.return_value = Mock()
            feature = CaptionFeature()
            
            assert feature.name == "caption"
    
    def test_description_property(self):
        """Test description property"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.return_value = Mock()
            feature = CaptionFeature()
            
            assert feature.description == "Image captioning using vision-language models"
    
    def test_process_frame_not_initialized(self):
        """Test process_frame when feature is not initialized"""
        feature = CaptionFeature()
        feature.initialized = False
        
        frame = self._create_test_frame()
        
        with pytest.raises(ProcessingError, match="Captioning feature not initialized"):
            feature.process_frame(frame)
    
    def test_process_frame_no_frame_data(self):
        """Test process_frame with no frame data"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.return_value = Mock()
            feature = CaptionFeature()
            
            frame = VideoFramePacket(
                frame_data=None,
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video"
            )
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, CaptionResult)
            assert result.caption == ""
    
    def test_process_frame_success(self):
        """Test successful frame processing"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner_instance = Mock()
            mock_captioner_instance.caption_image.return_value = "A beautiful sunset over the ocean"
            mock_captioner.return_value = mock_captioner_instance
            
            feature = CaptionFeature()
            frame = self._create_test_frame()
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, CaptionResult)
            assert result.caption == "A beautiful sunset over the ocean"
            assert result.model_name == "vikhyatk/moondream2"
            assert result.caption_length == "long"
    
    def test_process_frame_processing_error(self):
        """Test process_frame with processing error"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner_instance = Mock()
            mock_captioner_instance.caption_image.side_effect = Exception("Processing failed")
            mock_captioner.return_value = mock_captioner_instance
            
            feature = CaptionFeature()
            frame = self._create_test_frame()
            
            with pytest.raises(ProcessingError, match="Error in captioning processing"):
                feature.process_frame(frame)
    
    def test_process_batch_not_initialized(self):
        """Test process_batch when feature is not initialized"""
        feature = CaptionFeature()
        feature.initialized = False
        
        frames = [self._create_test_frame(i) for i in range(3)]
        
        with pytest.raises(ProcessingError, match="Captioning feature not initialized"):
            feature.process_batch(frames)
    
    def test_process_batch_no_valid_frames(self):
        """Test process_batch with no valid frames"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.return_value = Mock()
            feature = CaptionFeature()
            
            frames = [VideoFramePacket(
                frame_data=None,
                frame_number=i,
                timestamp=datetime.timedelta(seconds=i/30.0),
                source_video_id="test_video"
            ) for i in range(3)]
            
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, CaptionResult)
                assert result.caption == ""
    
    def test_process_batch_success(self):
        """Test successful batch processing"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner_instance = Mock()
            mock_captioner_instance.caption_image.side_effect = [
                "A beautiful sunset",
                "A mountain landscape",
                "A city skyline"
            ]
            mock_captioner.return_value = mock_captioner_instance
            
            feature = CaptionFeature()
            frames = [self._create_test_frame(i) for i in range(3)]
            
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            for i, result in enumerate(results):
                assert isinstance(result, CaptionResult)
                assert result.model_name == "vikhyatk/moondream2"
                assert result.caption_length == "long"
            
            assert results[0].caption == "A beautiful sunset"
            assert results[1].caption == "A mountain landscape"
            assert results[2].caption == "A city skyline"
    
    def test_process_batch_mixed_frames(self):
        """Test process_batch with mixed valid and invalid frames"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner_instance = Mock()
            mock_captioner_instance.caption_image.return_value = "A test image"
            mock_captioner.return_value = mock_captioner_instance
            
            feature = CaptionFeature()
            valid_frame = self._create_test_frame(0)
            invalid_frame = VideoFramePacket(
                frame_data=None,
                frame_number=1,
                timestamp=datetime.timedelta(seconds=1/30.0),
                source_video_id="test_video"
            )
            
            frames = [valid_frame, invalid_frame, valid_frame]
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            assert results[0].caption == "A test image"
            assert results[1].caption == ""
            assert results[2].caption == "A test image"
    
    def test_process_batch_processing_error(self):
        """Test process_batch with processing error"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner_instance = Mock()
            mock_captioner_instance.caption_image.side_effect = Exception("Batch processing failed")
            mock_captioner.return_value = mock_captioner_instance
            
            feature = CaptionFeature()
            frames = [self._create_test_frame(i) for i in range(3)]
            
            with pytest.raises(ProcessingError, match="Error in batch captioning processing"):
                feature.process_batch(frames)
    
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


class TestCaptionFeatureIntegration:
    """Integration tests for CaptionFeature"""
    
    def test_process_frame_with_real_image(self):
        """Test processing frame with real image data"""
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner_instance = Mock()
            mock_captioner_instance.caption_image.return_value = "A red square"
            mock_captioner.return_value = mock_captioner_instance
            
            feature = CaptionFeature()
            
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
            
            assert isinstance(result, CaptionResult)
            assert result.caption == "A red square"
    
    def test_config_parameter_access(self):
        """Test configuration parameter access"""
        config = {
            "model": "custom/model",
            "nested": {
                "param": "value"
            }
        }
        
        with patch('cortexia.features.caption.models.MoonDreamCaptioner') as mock_captioner:
            mock_captioner.return_value = Mock()
            feature = CaptionFeature(config)
            
            assert feature.get_config_param("model") == "custom/model"
            assert feature.get_config_param("nested.param") == "value"
            assert feature.get_config_param("nonexistent") is None
            assert feature.get_config_param("nonexistent", "default") == "default"