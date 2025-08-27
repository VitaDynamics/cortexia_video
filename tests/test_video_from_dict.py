#!/usr/bin/env python3
"""Dummy input for testing VideoFramePacket.from_dict function"""

import datetime
import numpy as np
import sys
import os

# Add the project root to the path so we can import cortexia
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cortexia.data.models.video import VideoFramePacket

def test_from_dict():
    """Test VideoFramePacket.from_dict with dummy data"""
    
    # Create a dummy dictionary that would come from JSON or other serialized source
    dummy_dict = {
        'frame_data': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # Random RGB frame
        'frame_number': 42,
        'timestamp': 10.5,  # seconds
        'source_video_id': 'test_video_001',
        'additional_metadata': {
            'fps': 30,
            'resolution': '640x480',
            'codec': 'h264'
        },
        'annotations': {
            'CaptionResult': 'A beautiful sunset over the ocean',
            'DetectionResult': {
                'boxes': [[10, 10, 50, 50], [100, 100, 150, 150]],
                'scores': [0.95, 0.87],
                'labels': ['person', 'car']
            }
        }
    }

    try:
        # Test the from_dict function
        frame_packet = VideoFramePacket.from_dict(dummy_dict)
        print("✓ Successfully created VideoFramePacket from dict")
        print(f"Frame number: {frame_packet.frame_number}")
        print(f"Timestamp: {frame_packet.timestamp}")
        print(f"Source video ID: {frame_packet.source_video_id}")
        print(f"Frame data shape: {frame_packet.frame_data.shape}")
        print(f"Frame data type: {frame_packet.frame_data.dtype}")
        print(f"Has annotations: {frame_packet.has_annotations()}")
        print(f"Additional metadata: {frame_packet.additional_metadata}")
        
        # Test round-trip conversion
        back_to_dict = frame_packet.dict()
        print("\n✓ Round-trip conversion successful")
        print(f"Original timestamp: {dummy_dict['timestamp']}")
        print(f"Converted back timestamp: {back_to_dict['timestamp']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_serialization():
    """Test the serialize/deserialize functionality"""
    
    # Create a VideoFramePacket
    frame_packet = VideoFramePacket(
        frame_data=np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
        frame_number=100,
        timestamp=datetime.timedelta(seconds=25.5),
        source_video_id='serialization_test',
        additional_metadata={'test': 'data'}
    )
    
    try:
        # Test serialization
        serialized = frame_packet.serialize()
        print("\n✓ Serialization successful")
        print(f"Serialized keys: {list(serialized.keys())}")
        
        # Test deserialization
        deserialized = VideoFramePacket.deserialize(serialized)
        print("✓ Deserialization successful")
        
        # Verify data integrity
        assert deserialized.frame_number == frame_packet.frame_number
        assert deserialized.timestamp == frame_packet.timestamp
        assert deserialized.source_video_id == frame_packet.source_video_id
        assert np.array_equal(deserialized.frame_data, frame_packet.frame_data)
        
        print("✓ Data integrity verified")
        return True
        
    except Exception as e:
        print(f"✗ Serialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing VideoFramePacket from_dict functionality...")
    
    success1 = test_from_dict()
    success2 = test_serialization()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)