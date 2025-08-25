"""Tests for segmentation feature"""

import pytest
import numpy as np
import datetime
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from cortexia.features.segmentation.segmenter import SegmentationFeature
from cortexia.data.models.video import VideoFramePacket
from cortexia.data.models.result.detection_result import BoundingBox, DetectionResult
from cortexia.data.models.result.segmentation_result import SegmentationResult
from cortexia.api.exceptions import ModelLoadError, ProcessingError


class MockVideoAnnotations:
    """Mock annotations for video frame packets"""
    
    def __init__(self, detections=None):
        self.detections = detections or []


class TestSegmentationFeature:
    """Test cases for SegmentationFeature"""
    
    def test_init_with_config(self):
        """Test initialization with configuration"""
        config = {
            "model": "facebook/sam-vit-large",
            "batch_size": 4
        }
        
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature(config)
            
            assert feature.config == config
            assert feature.segmenter is not None
            assert feature.device is not None
            assert feature.initialized is True
    
    def test_init_without_config(self):
        """Test initialization without configuration"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            assert feature.config == {}
            assert feature.initialized is True
    
    def test_initialize_success(self):
        """Test successful initialization"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            assert feature.initialized is True
            assert feature.segmenter is not None
    
    def test_initialize_failure(self):
        """Test initialization failure"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.side_effect = Exception("Model load failed")
            
            with pytest.raises(ModelLoadError, match="Failed to initialize segmentation model"):
                SegmentationFeature()
    
    def test_name_property(self):
        """Test name property"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            assert feature.name == "segmentation"
    
    def test_description_property(self):
        """Test description property"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            assert feature.description == "Object segmentation using Segment Anything Model (SAM)"
    
    def test_process_frame_not_initialized(self):
        """Test process_frame when feature is not initialized"""
        feature = SegmentationFeature()
        feature.initialized = False
        
        frame = self._create_test_frame()
        
        with pytest.raises(ProcessingError, match="Segmentation feature not initialized"):
            feature.process_frame(frame)
    
    def test_process_frame_no_frame_data(self):
        """Test process_frame with no frame data"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            frame = VideoFramePacket(
                frame_data=None,
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video"
            )
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, SegmentationResult)
            assert result.score == 0.0
            assert result.label == ""
            assert result.area == 0
            assert result.bbox.xmin == 0 and result.bbox.ymin == 0
    
    def test_process_frame_empty_frame_data(self):
        """Test process_frame with empty frame data"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            frame = VideoFramePacket(
                frame_data=np.array([]),
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video"
            )
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, SegmentationResult)
    
    def test_process_frame_no_detections(self):
        """Test process_frame with no detections in frame"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            frame = self._create_test_frame()
            frame.annotations = None
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, SegmentationResult)
            assert result.score == 0.0
            assert result.label == ""
            assert result.area == 0
    
    def test_process_frame_success_with_detections(self):
        """Test successful frame processing with detections"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter_instance = Mock()
            # Mock segmentation masks
            mock_mask = np.ones((100, 100), dtype=bool)
            mock_segmenter_instance.segment_object.return_value = [mock_mask]
            mock_segmenter.return_value = mock_segmenter_instance
            
            feature = SegmentationFeature()
            frame = self._create_test_frame_with_detections()
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, SegmentationResult)
            assert result.score > 0.0
            assert result.area > 0
            assert result.bbox.xmin >= 0
    
    def test_process_frame_processing_error(self):
        """Test process_frame with processing error"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter_instance = Mock()
            mock_segmenter_instance.segment_object.side_effect = Exception("Segmentation failed")
            mock_segmenter.return_value = mock_segmenter_instance
            
            feature = SegmentationFeature()
            frame = self._create_test_frame_with_detections()
            
            with pytest.raises(ProcessingError, match="Error in segmentation processing"):
                feature.process_frame(frame)
    
    def test_process_batch_not_initialized(self):
        """Test process_batch when feature is not initialized"""
        feature = SegmentationFeature()
        feature.initialized = False
        
        frames = [self._create_test_frame(i) for i in range(3)]
        
        with pytest.raises(ProcessingError, match="Segmentation feature not initialized"):
            feature.process_batch(frames)
    
    def test_process_batch_no_valid_frames(self):
        """Test process_batch with no valid frames"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            frames = [VideoFramePacket(
                frame_data=None,
                frame_number=i,
                timestamp=datetime.timedelta(seconds=i/30.0),
                source_video_id="test_video"
            ) for i in range(3)]
            
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, SegmentationResult)
                assert result.score == 0.0
                assert result.label == ""
                assert result.area == 0
    
    def test_process_batch_success(self):
        """Test successful batch processing"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter_instance = Mock()
            # Mock batch segmentation masks
            mock_masks = [
                [np.ones((50, 50), dtype=bool)],  # Frame 1
                [np.ones((60, 60), dtype=bool)],  # Frame 2
                [np.ones((70, 70), dtype=bool)]   # Frame 3
            ]
            mock_segmenter_instance.segment_object.return_value = mock_masks
            mock_segmenter.return_value = mock_segmenter_instance
            
            feature = SegmentationFeature()
            frames = [self._create_test_frame_with_detections(i) for i in range(3)]
            
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, SegmentationResult)
                assert result.area > 0
    
    def test_process_batch_mixed_frames(self):
        """Test process_batch with mixed valid and invalid frames"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter_instance = Mock()
            mock_masks = [
                [np.ones((50, 50), dtype=bool)],  # Valid frame 1
                [np.ones((60, 60), dtype=bool)]   # Valid frame 2 (3rd frame)
            ]
            mock_segmenter_instance.segment_object.return_value = mock_masks
            mock_segmenter.return_value = mock_segmenter_instance
            
            feature = SegmentationFeature()
            valid_frame = self._create_test_frame_with_detections(0)
            invalid_frame = VideoFramePacket(
                frame_data=None,
                frame_number=1,
                timestamp=datetime.timedelta(seconds=1/30.0),
                source_video_id="test_video"
            )
            
            frames = [valid_frame, invalid_frame, valid_frame]
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            assert results[0].area > 0  # Valid frame
            assert results[1].area == 0  # Invalid frame
            assert results[2].area > 0  # Valid frame
    
    def test_process_batch_processing_error(self):
        """Test process_batch with processing error"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter_instance = Mock()
            mock_segmenter_instance.segment_object.side_effect = Exception("Batch segmentation failed")
            mock_segmenter.return_value = mock_segmenter_instance
            
            feature = SegmentationFeature()
            frames = [self._create_test_frame_with_detections(i) for i in range(3)]
            
            with pytest.raises(ProcessingError, match="Error in batch segmentation processing"):
                feature.process_batch(frames)
    
    def test_create_segmentation_results(self):
        """Test _create_segmentation_results method"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            # Create test data
            masks = [
                np.ones((50, 50), dtype=bool),
                np.ones((60, 60), dtype=bool)
            ]
            boxes = [[10, 20, 50, 80], [60, 30, 90, 70]]
            detections = [
                DetectionResult(
                    score=0.9,
                    label="person",
                    box=BoundingBox(xmin=10, ymin=20, xmax=50, ymax=80),
                    id="det_1"
                ),
                DetectionResult(
                    score=0.8,
                    label="car",
                    box=BoundingBox(xmin=60, ymin=30, xmax=90, ymax=70),
                    id="det_2"
                )
            ]
            
            results = feature._create_segmentation_results(masks, boxes, detections)
            
            assert len(results) == 2
            
            assert results[0].area == 50 * 50
            assert results[0].bbox.xmin == 10
            assert results[0].bbox.ymin == 20
            assert results[0].detection_id == "det_1"
            
            assert results[1].area == 60 * 60
            assert results[1].bbox.xmin == 60
            assert results[1].bbox.ymin == 30
            assert results[1].detection_id == "det_2"
    
    def test_add_masks_to_detections(self):
        """Test _add_masks_to_detections method"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            # Create test data
            mask1 = np.ones((50, 50), dtype=bool)
            mask2 = np.ones((60, 60), dtype=bool)
            
            segments = [
                SegmentationResult(
                    mask=mask1,
                    score=1.0,
                    label="segment_1",
                    area=2500,
                    bbox=BoundingBox(xmin=10, ymin=20, xmax=50, ymax=80),
                    detection_id="det_1"
                ),
                SegmentationResult(
                    mask=mask2,
                    score=1.0,
                    label="segment_2",
                    area=3600,
                    bbox=BoundingBox(xmin=60, ymin=30, xmax=90, ymax=70),
                    detection_id="det_2"
                )
            ]
            
            detections = [
                Mock(id="det_1"),
                Mock(id="det_2"),
                Mock(id="det_3")  # No corresponding segment
            ]
            
            feature._add_masks_to_detections(segments, detections)
            
            assert np.array_equal(detections[0].mask, mask1)
            assert np.array_equal(detections[1].mask, mask2)
            assert not hasattr(detections[2], 'mask') or detections[2].mask is None
    
    def test_config_parameters(self):
        """Test configuration parameter access"""
        config = {
            "model": "facebook/sam-vit-huge",
            "batch_size": 16,
            "device": "cuda",
            "nested": {
                "param": "value"
            }
        }
        
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature(config)
            
            assert feature.get_config_param("model") == "facebook/sam-vit-huge"
            assert feature.get_config_param("batch_size") == 16
            assert feature.get_config_param("device") == "cuda"
            assert feature.get_config_param("nested.param") == "value"
            assert feature.get_config_param("nonexistent") is None
            assert feature.get_config_param("nonexistent", "default") == "default"
    
    def test_output_schema_validation(self):
        """Test output schema validation"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            assert feature.get_output_schema() == SegmentationResult
    
    def test_required_inputs_and_fields(self):
        """Test required inputs and fields attributes"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter.return_value = Mock()
            feature = SegmentationFeature()
            
            assert feature.required_inputs == []
            assert feature.required_fields == []
            assert feature.get_required_fields() == []
    
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
    
    def _create_test_frame_with_detections(self, frame_number=0):
        """Helper method to create test VideoFramePacket with detections"""
        frame = self._create_test_frame(frame_number)
        
        # Add mock detections
        detections = [
            DetectionResult(
                score=0.9,
                label="person",
                box=BoundingBox(xmin=10, ymin=20, xmax=50, ymax=80),
                id=f"det_{frame_number}_1"
            ),
            DetectionResult(
                score=0.8,
                label="car", 
                box=BoundingBox(xmin=60, ymin=30, xmax=90, ymax=70),
                id=f"det_{frame_number}_2"
            )
        ]
        
        frame.annotations = MockVideoAnnotations(detections)
        return frame


class TestSegmentationFeatureIntegration:
    """Integration tests for SegmentationFeature"""
    
    def test_process_frame_with_real_image(self):
        """Test processing frame with real image data"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            mock_segmenter_instance = Mock()
            mock_mask = np.zeros((224, 224), dtype=bool)
            mock_mask[50:150, 50:150] = True  # Square mask
            mock_segmenter_instance.segment_object.return_value = [mock_mask]
            mock_segmenter.return_value = mock_segmenter_instance
            
            feature = SegmentationFeature()
            
            # Create a realistic image frame
            frame_data = np.zeros((224, 224, 3), dtype=np.uint8)
            frame_data[50:150, 50:150, 0] = 255  # Red square
            
            # Add detection for the red square
            detection = DetectionResult(
                score=0.9,
                label="red_square",
                box=BoundingBox(xmin=50, ymin=50, xmax=150, ymax=150),
                id="det_1"
            )
            
            frame = VideoFramePacket(
                frame_data=frame_data,
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video"
            )
            frame.annotations = MockVideoAnnotations([detection])
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, SegmentationResult)
            assert result.area == 100 * 100  # 100x100 square
            assert result.bbox.xmin == 50
            assert result.bbox.ymin == 50
            assert result.bbox.xmax == 150
            assert result.bbox.ymax == 150
    
    def test_device_initialization(self):
        """Test device initialization based on CUDA availability"""
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            with patch('torch.cuda.is_available', return_value=True):
                mock_segmenter.return_value = Mock()
                feature = SegmentationFeature()
                
                assert str(feature.device) == "cuda"
        
        with patch('cortexia.features.segmentation.models.ObjectSegmenter') as mock_segmenter:
            with patch('torch.cuda.is_available', return_value=False):
                mock_segmenter.return_value = Mock()
                feature = SegmentationFeature()
                
                assert str(feature.device) == "cpu"