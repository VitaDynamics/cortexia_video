"""Tests for listing feature"""

import pytest
import numpy as np
import datetime
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from cortexia.features.listing.lister import ListingFeature
from cortexia.data.models.video import VideoFramePacket
from cortexia.data.models.result.tagging_result import TaggingResult
from cortexia.api.exceptions import ModelLoadError, ProcessingError


class TestListingFeature:
    """Test cases for ListingFeature"""
    
    def test_init_with_config(self):
        """Test initialization with configuration"""
        config = {
            "model": "test/model",
            "custom_param": "test_value"
        }
        
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature(config)
            
            assert feature.config == config
            assert feature.lister is not None
    
    def test_init_without_config(self):
        """Test initialization without configuration"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature()
            
            assert feature.config == {}
    
    def test_initialize_success(self):
        """Test successful initialization"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature()
            
            assert feature.initialized is True
            assert feature.lister is not None
    
    def test_initialize_failure(self):
        """Test initialization failure"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.side_effect = Exception("Model load failed")
            
            with pytest.raises(ModelLoadError, match="Failed to initialize listing model"):
                ListingFeature()
    
    def test_name_property(self):
        """Test name property"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature()
            
            assert feature.name == "listing"
    
    def test_description_property(self):
        """Test description property"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature()
            
            assert feature.description == "Object listing using various vision-language models"
    
    def test_process_frame_not_initialized(self):
        """Test process_frame when feature is not initialized"""
        feature = ListingFeature()
        feature.initialized = False
        
        frame = self._create_test_frame()
        
        with pytest.raises(ProcessingError, match="Listing feature not initialized"):
            feature.process_frame(frame)
    
    def test_process_frame_no_frame_data(self):
        """Test process_frame with no frame data"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature()
            
            frame = VideoFramePacket(
                frame_data=None,
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video"
            )
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, TaggingResult)
            assert result.tags == []
    
    def test_process_frame_success(self):
        """Test successful frame processing"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.return_value = ["car", "person", "tree"]
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
            frame = self._create_test_frame()
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, TaggingResult)
            assert result.tags == ["car", "person", "tree"]
            assert result.model_name == "vikhyatk/moondream2"
            assert result.raw_response == "car. person. tree."
    
    def test_process_frame_empty_objects(self):
        """Test process_frame with empty objects list"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.return_value = []
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
            frame = self._create_test_frame()
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, TaggingResult)
            assert result.tags == []
            assert result.raw_response == ""
    
    def test_process_frame_processing_error(self):
        """Test process_frame with processing error"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.side_effect = Exception("Processing failed")
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
            frame = self._create_test_frame()
            
            with pytest.raises(ProcessingError, match="Error in listing processing"):
                feature.process_frame(frame)
    
    def test_process_batch_not_initialized(self):
        """Test process_batch when feature is not initialized"""
        feature = ListingFeature()
        feature.initialized = False
        
        frames = [self._create_test_frame(i) for i in range(3)]
        
        with pytest.raises(ProcessingError, match="Listing feature not initialized"):
            feature.process_batch(frames)
    
    def test_process_batch_no_valid_frames(self):
        """Test process_batch with no valid frames"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature()
            
            frames = [VideoFramePacket(
                frame_data=None,
                frame_number=i,
                timestamp=datetime.timedelta(seconds=i/30.0),
                source_video_id="test_video"
            ) for i in range(3)]
            
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, TaggingResult)
                assert result.tags == []
    
    def test_process_batch_success(self):
        """Test successful batch processing"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.side_effect = [
                ["car", "person"],
                ["tree", "building"],
                ["cat", "dog"]
            ]
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
            frames = [self._create_test_frame(i) for i in range(3)]
            
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            assert results[0].tags == ["car", "person"]
            assert results[1].tags == ["tree", "building"]
            assert results[2].tags == ["cat", "dog"]
            
            for result in results:
                assert isinstance(result, TaggingResult)
                assert result.model_name == "vikhyatk/moondream2"
    
    def test_process_batch_mixed_frames(self):
        """Test process_batch with mixed valid and invalid frames"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.return_value = ["test_object"]
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
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
            assert results[0].tags == ["test_object"]
            assert results[1].tags == []
            assert results[2].tags == ["test_object"]
    
    def test_process_batch_processing_error(self):
        """Test process_batch with processing error"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.side_effect = Exception("Batch processing failed")
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
            frames = [self._create_test_frame(i) for i in range(3)]
            
            with pytest.raises(ProcessingError, match="Error in batch listing processing"):
                feature.process_batch(frames)
    
    def test_list_objects_private_method(self):
        """Test the private _list_objects method"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.return_value = ["test_object"]
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
            image = Image.new('RGB', (100, 100), color='red')
            
            result = feature._list_objects(image)
            
            assert result == ["test_object"]
            mock_lister_instance.list_objects_in_image.assert_called_once_with(image)
    
    def test_list_objects_batch_private_method(self):
        """Test the private _list_objects_batch method"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.side_effect = [
                ["object1"], ["object2"], ["object3"]
            ]
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
            images = [Image.new('RGB', (100, 100), color='red') for _ in range(3)]
            
            results = feature._list_objects_batch(images)
            
            assert len(results) == 3
            assert results == [["object1"], ["object2"], ["object3"]]
            assert mock_lister_instance.list_objects_in_image.call_count == 3
    
    def test_config_parameters(self):
        """Test configuration parameter access"""
        config = {
            "model": "custom/model",
            "nested": {
                "param": "value"
            }
        }
        
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature(config)
            
            assert feature.get_config_param("model") == "custom/model"
            assert feature.get_config_param("nested.param") == "value"
            assert feature.get_config_param("nonexistent") is None
            assert feature.get_config_param("nonexistent", "default") == "default"
    
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


class TestListingFeatureIntegration:
    """Integration tests for ListingFeature"""
    
    def test_process_frame_with_real_image(self):
        """Test processing frame with real image data"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister_instance = Mock()
            mock_lister_instance.list_objects_in_image.return_value = ["red_square"]
            mock_lister.return_value = mock_lister_instance
            
            feature = ListingFeature()
            
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
            
            assert isinstance(result, TaggingResult)
            assert result.tags == ["red_square"]
    
    def test_feature_registry_registration(self):
        """Test that the feature is properly registered"""
        from cortexia.features.registry import feature_registry
        
        # Check if listing feature is registered
        assert "listing" in feature_registry._registry
        assert feature_registry._registry["listing"] == ListingFeature
    
    def test_output_schema_validation(self):
        """Test output schema validation"""
        with patch('cortexia.features.listing.models.MoonDreamLister') as mock_lister:
            mock_lister.return_value = Mock()
            feature = ListingFeature()
            
            assert feature.get_output_schema() == TaggingResult
            assert hasattr(feature, 'output_schema')
            assert feature.output_schema == TaggingResult