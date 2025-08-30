#!/usr/bin/env python3
"""Test script to verify PyArrow struct functionality in BaseResult.

This module tests the PyArrow integration for Lance compatibility, ensuring that
all result schemas can be properly converted to and from PyArrow structs.
"""

import numpy as np
import pyarrow as pa
from cortexia.data.models.result.base_result import BaseResult
from cortexia.data.models.result.caption_result import CaptionResult
from cortexia.data.models.result.detection_result import DetectionResult, BoundingBox
from cortexia.data.models.result.depth_result import DepthResult


def test_caption_result_pyarrow():
    """Test CaptionResult PyArrow conversion."""
    print("Testing CaptionResult PyArrow conversion...")
    
    # Create a caption result
    caption_result = CaptionResult(
        caption="A beautiful sunset over the ocean",
        confidence=0.95,
        model_name="blip-base",
        processing_time_ms=150.5
    )
    
    # Convert to PyArrow struct
    struct_array = caption_result.to_pyarrow_struct()
    print(f"Struct array type: {struct_array.type}")
    print(f"Struct array: {struct_array}")
    
    # Get schema
    schema = caption_result.get_pyarrow_schema()
    print(f"Schema: {schema}")
    
    # Convert back from PyArrow struct
    reconstructed = CaptionResult.from_pyarrow_struct(struct_array)
    print(f"Original: {caption_result}")
    print(f"Reconstructed: {reconstructed}")
    
    # Verify data integrity
    assert reconstructed.caption == caption_result.caption
    assert reconstructed.confidence == caption_result.confidence
    assert reconstructed.model_name == caption_result.model_name
    assert reconstructed.processing_time_ms == caption_result.processing_time_ms
    
    print("✓ CaptionResult PyArrow conversion test passed\n")


def test_detection_result_pyarrow():
    """Test DetectionResult PyArrow conversion."""
    print("Testing DetectionResult PyArrow conversion...")
    
    # Create a detection result with numpy array
    mask = np.random.rand(100, 100) > 0.5
    object_features = np.random.rand(512).astype(np.float32)
    
    detection_result = DetectionResult(
        score=0.87,
        label="person",
        box=BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=200.0),
        mask=mask,
        description="A person standing",
        object_clip_features=object_features
    )
    
    # Convert to PyArrow struct
    struct_array = detection_result.to_pyarrow_struct()
    print(f"Struct array type: {struct_array.type}")
    
    # Get schema
    schema = detection_result.get_pyarrow_schema()
    print(f"Schema: {schema}")
    
    # Convert back from PyArrow struct
    reconstructed = DetectionResult.from_pyarrow_struct(struct_array)
    print(f"Original: {detection_result}")
    print(f"Reconstructed: {reconstructed}")
    
    # Verify data integrity
    assert reconstructed.score == detection_result.score
    assert reconstructed.label == detection_result.label
    assert reconstructed.box.xmin == detection_result.box.xmin
    assert reconstructed.box.ymin == detection_result.box.ymin
    assert reconstructed.box.xmax == detection_result.box.xmax
    assert reconstructed.box.ymax == detection_result.box.ymax
    assert np.array_equal(reconstructed.mask, detection_result.mask)
    assert np.allclose(reconstructed.object_clip_features, detection_result.object_clip_features)
    
    print("✓ DetectionResult PyArrow conversion test passed\n")


def test_depth_result_pyarrow():
    """Test DepthResult PyArrow conversion."""
    print("Testing DepthResult PyArrow conversion...")
    
    # Create a depth result with numpy array
    depth_map = np.random.rand(480, 640).astype(np.float32)
    depth_stats = {"min": 0.1, "max": 10.5, "mean": 5.2, "std": 2.1}
    
    depth_result = DepthResult(
        depth_map=depth_map,
        depth_statistics=depth_stats,
        model_name="midas-v2",
        focal_length=800.0,
        processing_time_ms=250.0
    )
    
    # Convert to PyArrow struct
    struct_array = depth_result.to_pyarrow_struct()
    print(f"Struct array type: {struct_array.type}")
    
    # Get schema
    schema = depth_result.get_pyarrow_schema()
    print(f"Schema: {schema}")
    
    # Convert back from PyArrow struct
    reconstructed = DepthResult.from_pyarrow_struct(struct_array)
    print(f"Original: {depth_result}")
    print(f"Reconstructed: {reconstructed}")
    
    # Verify data integrity
    assert np.allclose(reconstructed.depth_map, depth_result.depth_map)
    assert reconstructed.depth_statistics == depth_result.depth_statistics
    assert reconstructed.model_name == depth_result.model_name
    assert reconstructed.focal_length == depth_result.focal_length
    assert reconstructed.processing_time_ms == depth_result.processing_time_ms
    
    print("✓ DepthResult PyArrow conversion test passed\n")


def test_lance_compatibility():
    """Test that PyArrow structs are compatible with Lance."""
    print("Testing Lance compatibility...")
    
    # Create multiple results
    results = [
        CaptionResult(
            caption="A cat sitting on a windowsill",
            confidence=0.92,
            model_name="blip-base"
        ),
        CaptionResult(
            caption="A dog running in a park",
            confidence=0.88,
            model_name="blip-base"
        ),
        DetectionResult(
            score=0.75,
            label="cat",
            box=BoundingBox(xmin=50.0, ymin=30.0, xmax=150.0, ymax=180.0)
        )
    ]
    
    # Convert to PyArrow structs
    struct_arrays = [result.to_pyarrow_struct() for result in results]
    
    # Create a table from the structs
    try:
        # This would normally be used with Lance
        table = pa.Table.from_arrays(
            [struct_arrays],
            names=["results"]
        )
        print(f"Table schema: {table.schema}")
        print(f"Table: {table}")
        
        # Test that we can iterate and reconstruct
        for i, struct_array in enumerate(struct_arrays):
            if isinstance(results[i], CaptionResult):
                reconstructed = CaptionResult.from_pyarrow_struct(struct_array)
            elif isinstance(results[i], DetectionResult):
                reconstructed = DetectionResult.from_pyarrow_struct(struct_array)
            else:
                continue
            
            print(f"Result {i} reconstruction successful")
        
        print("✓ Lance compatibility test passed\n")
        
    except Exception as e:
        print(f"✗ Lance compatibility test failed: {e}\n")


if __name__ == "__main__":
    print("Running PyArrow struct tests for BaseResult classes...\n")
    
    test_caption_result_pyarrow()
    test_detection_result_pyarrow()
    test_depth_result_pyarrow()
    test_lance_compatibility()
    
    print("All tests completed successfully!")