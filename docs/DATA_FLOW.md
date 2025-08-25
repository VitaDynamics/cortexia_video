# Cortexia Video Data Flow Architecture

This document describes the unified data flow architecture for the Cortexia Video processing pipeline, detailing how video frames move through annotation, gating, and buffering stages.

## Overview

The Cortexia Video SDK implements a unified data flow pipeline that progressively enriches video frames with annotations and gate decisions. The architecture supports three main processing stages:

1. **Feature Extraction** - Annotates frames with AI-powered analysis
2. **Gate Processing** - Applies filtering criteria and quality checks
3. **Buffer Management** - Manages frame collections with sliding window logic

## Unified Data Flow Pipeline

```
VideoFramePacket � Features � AnnotatedFramePacket � Gates � TaggedFramePacket � Buffer
```

### Stage 1: Raw Video Input
- **Input**: `VideoFramePacket`
- **Contains**: Raw frame data, metadata, timestamps
- **Purpose**: Base container for video frame information

### Stage 2: Feature Annotation
- **Input**: `VideoFramePacket`
- **Output**: `AnnotatedFramePacket`
- **Process**: Features analyze frame content and add annotations
- **Contains**: Original frame + annotation results (detections, segmentations, etc.)

### Stage 3: Gate Processing
- **Input**: `AnnotatedFramePacket` (or any frame type)
- **Output**: `TaggedFramePacket`
- **Process**: Gates apply filtering criteria and quality checks
- **Contains**: Original frame + annotations + gate decisions/scores

### Stage 4: Buffer Management
- **Input**: `TaggedFramePacket` (or any frame type)
- **Process**: Manages frame collections using sliding window algorithms
- **Purpose**: Temporal frame management and selection

## Data Models

Check @cortexia/data/models/video.py


#### AnnotationResults
Contains structured results from feature extraction:
- Object detections
- Semantic segmentations
- Feature embeddings
- Classification results

## Component Architecture

### BaseFeature
- **Purpose**: Extracts annotations from video frames
- **Input**: `VideoFramePacket`
- **Output**: Annotation data (integrated into `AnnotatedFramePacket`)
- **Key Methods**:
  - `process_frame()` - Single frame processing
  - `process_batch()` - Batch processing for efficiency

### BaseGate
- **Purpose**: Filters frames based on quality criteria
- **Input**: `FrameInput = Union[VideoFramePacket, AnnotatedFramePacket, TaggedFramePacket]`
- **Output**: Boolean decisions + optional metadata
- **Key Methods**:
  - `process_frame()` - Core filtering logic
  - `process_with_metadata()` - Returns detailed results
  - `process_to_tagged()` - Converts to TaggedFramePacket

### BaseBuffer
- **Purpose**: Manages frame collections with sliding window logic
- **Input**: `BufferInput = Union[VideoFramePacket, AnnotatedFramePacket, TaggedFramePacket]`
- **Features**:
  - Automatic oldest frame eviction
  - Configurable window sizes
  - Smart frame ID extraction


### Gate Decision Access
```python
# Check if frame passed all gates
if tagged_frame.passes_gates:
    process_frame(tagged_frame)

# Access individual gate results
blur_passed = tagged_frame.gate_results.gate_decisions.get('BlurGate', True)
blur_score = tagged_frame.gate_results.gate_scores.get('BlurGate', 0.0)
```

### Buffer Utilization
```python
# Smart buffering can utilize gate results
buffer = SlidingTopKBuffer(window_size=10, filter_failed_gates=True)
buffer.add(tagged_frame)  # Automatically filters based on gate_results.passes

# Access buffered frames with full context
window_frames = buffer.get_window()  # List[TaggedFramePacket]
```

## Benefits of Unified Architecture

### Schema Consistency
- Standardized data models across all pipeline stages
- Predictable interfaces for component integration
- Simplified debugging and monitoring

### Backward Compatibility
- Existing gates and buffers continue to work unchanged
- Gradual migration path for legacy code
- Type unions support multiple frame formats

### Enhanced Metadata Tracking
- Comprehensive gate decision history
- Confidence scores and metadata preservation
- Full annotation context availability

### Flexible Processing
- Gates can access both raw frames and annotations
- Buffers can make intelligent decisions based on gate results
- Pipeline stages can be reordered or skipped as needed

### Performance Optimization
- Batch processing support in features
- Efficient frame type detection
- Minimal overhead for type conversions

## Implementation Notes

### Type Safety
The pipeline uses Union types to maintain flexibility while providing type safety:
```python
FrameInput = Union[VideoFramePacket, AnnotatedFramePacket, TaggedFramePacket]
BufferInput = Union[VideoFramePacket, AnnotatedFramePacket, TaggedFramePacket]
```

### Backward Compatibility
All existing code continues to work:
- Gates accepting `VideoFramePacket` automatically support the new types
- Buffers handle any frame type through smart property access
- Legacy pipelines require no modifications

### Error Handling
- Type checking at runtime ensures data integrity
- Graceful degradation when optional data is missing
- Comprehensive logging for debugging pipeline issues

## Future Extensions

The unified architecture supports future enhancements:
- Multi-modal annotations (audio, text metadata)
- Hierarchical gate structures
- Dynamic buffer strategies based on content analysis
- Cross-frame temporal analysis in gates