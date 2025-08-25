"""Tests for detection feature"""

import pytest
import numpy as np
import datetime
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from cortexia.features.detection.detector import DetectionFeature
from cortexia.data.models.video import VideoFramePacket
from cortexia.data.models.result.detection_result import BoundingBox, DetectionResult
from cortexia.api.exceptions import ModelLoadError, ProcessingError


class TestDetectionFeature:
    """Test cases for DetectionFeature"""
    
    def test_init_with_config(self):
        """Test initialization with configuration"""
        config = {
            "model": "test/model",
            "box_threshold": 0.5,
            "text_threshold": 0.4
        }
        
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature(config)
            
            assert feature.config == config
            assert feature.detector is not None
            assert feature.device is not None
    
    def test_init_without_config(self):
        """Test initialization without configuration"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            assert feature.config == {}
    
    def test_initialize_success(self):
        """Test successful initialization"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            assert feature.initialized is True
            assert feature.detector is not None
    
    def test_initialize_failure(self):
        """Test initialization failure"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.side_effect = Exception("Model load failed")
            
            with pytest.raises(ModelLoadError, match="Failed to initialize detection model"):
                DetectionFeature()
    
    def test_name_property(self):
        """Test name property"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            assert feature.name == "detection"
    
    def test_description_property(self):
        """Test description property"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            assert feature.description == "Object detection using Grounding DINO model"
    
    def test_process_frame_not_initialized(self):
        """Test process_frame when feature is not initialized"""
        feature = DetectionFeature()
        feature.initialized = False
        
        frame = self._create_test_frame()
        
        with pytest.raises(ProcessingError, match="Detection feature not initialized"):
            feature.process_frame(frame)
    
    def test_process_frame_no_frame_data(self):
        """Test process_frame with no frame data"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            frame = VideoFramePacket(
                frame_data=None,
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video"
            )
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, DetectionResult)
            assert result.score == 0.0
            assert result.label == ""
            assert result.box.xmin == 0 and result.box.ymin == 0
    
    def test_process_frame_empty_frame_data(self):
        """Test process_frame with empty frame data"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            frame = VideoFramePacket(
                frame_data=np.array([]),
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video"
            )
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, DetectionResult)
            assert result.score == 0.0
            assert result.label == ""
    
    def test_process_frame_success(self):
        """Test successful frame processing"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect_objects.return_value = [[
                {
                    'score': 0.95,
                    'label': 'person',
                    'box': [10, 20, 50, 80]
                },
                {
                    'score': 0.85,
                    'label': 'car',
                    'box': [60, 30, 90, 70]
                }
            ]]
            mock_detector.return_value = mock_detector_instance
            
            feature = DetectionFeature()
            frame = self._create_test_frame()
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, DetectionResult)
            assert result.score == 0.95
            assert result.label == "person"
            assert result.box.xmin == 10
            assert result.box.ymin == 20
            assert result.box.xmax == 50
            assert result.box.ymax == 80
    
    def test_process_frame_no_detections(self):
        """Test process_frame with no detections"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect_objects.return_value = [[]]
            mock_detector.return_value = mock_detector_instance
            
            feature = DetectionFeature()
            frame = self._create_test_frame()
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, DetectionResult)
            assert result.score == 0.0
            assert result.label == ""
    
    def test_process_frame_processing_error(self):
        """Test process_frame with processing error"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect_objects.side_effect = Exception("Processing failed")
            mock_detector.return_value = mock_detector_instance
            
            feature = DetectionFeature()
            frame = self._create_test_frame()
            
            with pytest.raises(ProcessingError, match="Error in detection processing"):
                feature.process_frame(frame)
    
    def test_process_batch_not_initialized(self):
        """Test process_batch when feature is not initialized"""
        feature = DetectionFeature()
        feature.initialized = False
        
        frames = [self._create_test_frame(i) for i in range(3)]
        
        with pytest.raises(ProcessingError, match="Detection feature not initialized"):
            feature.process_batch(frames)
    
    def test_process_batch_no_valid_frames(self):
        """Test process_batch with no valid frames"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            frames = [VideoFramePacket(
                frame_data=None,
                frame_number=i,
                timestamp=datetime.timedelta(seconds=i/30.0),
                source_video_id="test_video"
            ) for i in range(3)]
            
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, DetectionResult)
                assert result.score == 0.0
                assert result.label == ""
    
    def test_process_batch_success(self):
        """Test successful batch processing"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect_objects.return_value = [
                [{'score': 0.9, 'label': 'person', 'box': [10, 20, 50, 80]}],
                [{'score': 0.8, 'label': 'car', 'box': [60, 30, 90, 70]}],
                [{'score': 0.7, 'label': 'tree', 'box': [100, 10, 130, 90]}]
            ]
            mock_detector.return_value = mock_detector_instance
            
            feature = DetectionFeature()
            frames = [self._create_test_frame(i) for i in range(3)]
            
            results = feature.process_batch(frames)
            
            assert len(results) == 3
            assert results[0].label == "person"
            assert results[1].label == "car"
            assert results[2].label == "tree"
            
            for result in results:
                assert isinstance(result, DetectionResult)
                assert result.score > 0.0
    
    def test_process_batch_mixed_frames(self):
        """Test process_batch with mixed valid and invalid frames"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect_objects.return_value = [
                [{'score': 0.9, 'label': 'person', 'box': [10, 20, 50, 80]}],
                [{'score': 0.8, 'label': 'car', 'box': [60, 30, 90, 70]}]
            ]
            mock_detector.return_value = mock_detector_instance
            
            feature = DetectionFeature()
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
            assert results[0].label == "person"
            assert results[1].score == 0.0
            assert results[2].label == "car"
    
    def test_process_batch_processing_error(self):
        """Test process_batch with processing error"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect_objects.side_effect = Exception("Batch processing failed")
            mock_detector.return_value = mock_detector_instance
            
            feature = DetectionFeature()
            frames = [self._create_test_frame(i) for i in range(3)]
            
            with pytest.raises(ProcessingError, match="Error in batch detection processing"):
                feature.process_batch(frames)
    
    def test_get_detection_prompts_lister_results(self):
        """Test _get_detection_prompts with lister results in metadata"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            frame = self._create_test_frame()
            frame.additional_metadata = {
                'lister_results': ['car', 'person', 'tree']
            }
            
            prompts = feature._get_detection_prompts(frame)
            
            assert prompts == ['car', 'person', 'tree']
    
    def test_get_detection_prompts_dino_prompt(self):
        """Test _get_detection_prompts with dino prompt in metadata"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            frame = self._create_test_frame()
            frame.additional_metadata = {
                'dino_prompt': 'car.person.tree'
            }
            
            prompts = feature._get_detection_prompts(frame)
            
            assert prompts == ['car', 'person', 'tree']
    
    def test_get_detection_prompts_default(self):
        """Test _get_detection_prompts with default prompts"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            frame = self._create_test_frame()
            frame.additional_metadata = {}
            
            prompts = feature._get_detection_prompts(frame)
            
            assert prompts == ["object"]
    
    def test_get_detection_prompts_custom_default(self):
        """Test _get_detection_prompts with custom default prompts from config"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            config = {"default_prompts": ["vehicle", "person"]}
            feature = DetectionFeature(config)
            
            frame = self._create_test_frame()
            frame.additional_metadata = {}
            
            prompts = feature._get_detection_prompts(frame)
            
            assert prompts == ["vehicle", "person"]
    
    def test_convert_to_detection_results(self):
        """Test _convert_to_detection_results method"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            detection_dicts = [
                {'score': 0.9, 'label': 'person', 'box': [10, 20, 50, 80]},
                {'score': 0.8, 'label': 'car', 'box': [60, 30, 90, 70]}
            ]
            
            results = feature._convert_to_detection_results(detection_dicts)
            
            assert len(results) == 2
            
            assert results[0].score == 0.9
            assert results[0].label == "person"
            assert results[0].box.xmin == 10
            assert results[0].box.ymin == 20
            assert results[0].box.xmax == 50
            assert results[0].box.ymax == 80
            
            assert results[1].score == 0.8
            assert results[1].label == "car"
            assert results[1].box.xmin == 60
            assert results[1].box.ymin == 30
            assert results[1].box.xmax == 90
            assert results[1].box.ymax == 70
    
    def test_config_parameters(self):
        """Test configuration parameter access"""
        config = {
            "model": "custom/model",
            "box_threshold": 0.5,
            "text_threshold": 0.4,
            "nested": {
                "param": "value"
            }
        }
        
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature(config)
            
            assert feature.get_config_param("model") == "custom/model"
            assert feature.get_config_param("box_threshold") == 0.5
            assert feature.get_config_param("text_threshold") == 0.4
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
            source_video_id="test_video",
            additional_metadata={}
        )


class TestDetectionFeatureIntegration:
    """Integration tests for DetectionFeature"""
    
    def test_process_frame_with_real_image(self):
        """Test processing frame with real image data"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect_objects.return_value = [[
                {'score': 0.9, 'label': 'red_square', 'box': [50, 50, 150, 150]}
            ]]
            mock_detector.return_value = mock_detector_instance
            
            feature = DetectionFeature()
            
            # Create a realistic image frame
            frame_data = np.zeros((224, 224, 3), dtype=np.uint8)
            frame_data[50:150, 50:150, 0] = 255  # Red square
            
            frame = VideoFramePacket(
                frame_data=frame_data,
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video",
                additional_metadata={}
            )
            
            result = feature.process_frame(frame)
            
            assert isinstance(result, DetectionResult)
            assert result.label == "red_square"
            assert result.score == 0.9
    
    def test_output_schema_validation(self):
        """Test output schema validation"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            assert feature.get_output_schema() == DetectionResult
            assert hasattr(feature, 'output_schema')
            assert feature.output_schema == DetectionResult
    
    def test_required_inputs_and_fields(self):
        """Test required inputs and fields attributes"""
        with patch('cortexia.features.detection.models.ObjectDetector') as mock_detector:
            mock_detector.return_value = Mock()
            feature = DetectionFeature()
            
            assert feature.required_inputs == []
            assert feature.required_fields == []
            assert feature.get_required_fields() == []