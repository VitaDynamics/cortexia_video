# Plan for Implementing Batch Processing in Video Pipeline

**High-Level Goal:** Modify the video processing pipeline to handle frames in configurable batches, decouple metadata and image data processing within these batches, and maintain the existing JSON/pickle logging format.

## Phase 1: Core Batching Infrastructure & Batched Component Integration

1.  **Configuration Update:**
    *   Introduce a `batch_size` parameter under the `[processing]` section in configuration files (e.g., `config/config.toml`).
    *   Update `ConfigManager` in [`cortexia_video/config_manager.py`](cortexia_video/config_manager.py) to read and provide this `batch_size`.
    *   In [`ProcessingManager`](cortexia_video/processing_manager.py), retrieve this `batch_size`.

2.  **Batch Frame Loading (`cortexia_video/data_io.py`):**
    *   Complete the batching logic in the [`load_video_frames`](cortexia_video/data_io.py:12) generator.
    *   It should yield a tuple: `(batch_frames_meta, batch_frames_np)`.
        *   `batch_frames_meta`: A list of dictionaries, where each dictionary contains `{'frame_number': int, 'timestamp': float}` for a frame.
        *   `batch_frames_np`: A list of `np.ndarray` objects, where each array is the image data for a frame.
    *   Example structure for yielding:
        ```python
        # In load_video_frames (inside the loop for batch_size > 1)
        # Ensure fps is available (e.g., fps = vr.get_avg_fps())
        batch_frames_meta_list = [] # Initialize list for metadata
        batch_frames_np_list = []   # Initialize list for numpy arrays
        for frame_count in frame_indices: # Assuming frame_indices is already defined
            frame_np = vr[frame_count].asnumpy()
            timestamp = frame_count / fps # Make sure fps is defined
            batch_frames_meta_list.append({'frame_number': frame_count, 'timestamp': timestamp})
            batch_frames_np_list.append(frame_np)
            if len(batch_frames_np_list) == batch_size:
                yield batch_frames_meta_list, batch_frames_np_list
                batch_frames_meta_list = [] # Reset for next batch
                batch_frames_np_list = []   # Reset for next batch
        # After loop, yield any remaining frames
        if batch_frames_np_list: # Check if lists are not empty before yielding
            yield batch_frames_meta_list, batch_frames_np_list
        ```

3.  **Modify `ProcessingManager.process_video` (`cortexia_video/processing_manager.py`):**
    *   The main processing loop (currently around [`line 106`](cortexia_video/processing_manager.py:106)) will iterate over `(batch_frames_meta, batch_frames_np)` yielded by the updated [`load_video_frames`](cortexia_video/data_io.py:12).
    *   For each batch:
        *   Initialize a list of `FrameData` Pydantic objects using `batch_frames_meta` (for `frame_number`, `timestamp`) and `batch_frames_np` (for `rgb_image`).
        *   Maintain a corresponding list of `PIL.Image` objects, converted from `batch_frames_np`, for components that expect PIL input.

4.  **Adapt Processing Component Methods for Batch Input:**
    *   **`ObjectLister` (e.g., `RAMLister.list_objects_in_image` in [`cortexia_video/object_listing.py`](cortexia_video/object_listing.py)):**
        *   **Action**: Refactor method to accept a batch of images (e.g., `images_data: List[Image.Image]` or a stacked `torch.Tensor`).
        *   It should transform each image, stack them into a batch tensor if necessary, pass the batch to the underlying model, and process batched results.
        *   Return type: `List[List[str]]` (a list of object name lists, one for each image in the input batch).
        *   **In `ProcessingManager`**: Call with the batch of PIL images. Populate `lister_results` for each `FrameData` in the batch and generate corresponding `dino_prompt`s.

    *   **`ObjectDetector.detect_objects` (in [`cortexia_video/object_detection.py`](cortexia_video/object_detection.py)):**
        *   **Action**: Modify method signature to `detect_objects(self, images_data: List[Image.Image], text_prompts: List[List[str]], ...)` (or `List[str]` if prompts are shared per image).
        *   The Hugging Face `self.processor` ([`line 53`](cortexia_video/object_detection.py:53)) should be called with `images=batch_pil_images` and `text=batch_dino_prompts`. Ensure appropriate `padding` and `truncation` are handled for text prompts if necessary.
        *   Adjust `target_sizes` for the batch.
        *   The method will return a list of detection dictionaries/results (one for each image in the batch).
        *   **In `ProcessingManager`**: Iterate through the batched detector output, create `DetectionResult` objects, and assign them to the correct `FrameData` instance in the current batch.

    *   **`ObjectSegmenter.segment_object` (in [`cortexia_video/object_segmentation.py`](cortexia_video/object_segmentation.py)):**
        *   **Action**: Modify method signature to accept `images_data: List[Image.Image]` and `batch_input_boxes: List[List[List[float]]]` (batch -> image -> boxes for that image).
        *   The `self.processor` call ([`line 51`](cortexia_video/object_segmentation.py:51)) should be updated to handle these batched `images` and `input_boxes`.
        *   Post-processing of `outputs.pred_masks` and `outputs.iou_scores` ([`lines 64-74`](cortexia_video/object_segmentation.py:64-74)) must be adapted to iterate over batched predictions.
        *   Return type: `List[List[np.ndarray]]` (a list of lists of masks; outer list for batch, inner for detections in an image).
        *   **In `ProcessingManager`**: After detection, prepare `batch_input_boxes` by collecting all `detection.box.xyxy` from all `FrameData` objects in the current batch. Call the updated `segment_object`. Iterate through the batched segmenter output and create/update `SegmentationResult` objects, linking them correctly.

    *   **`FeatureExtractor` (e.g., `CLIPFeatureExtractor.extract_image_features` in [`cortexia_video/clip_wrapper.py`](cortexia_video/clip_wrapper.py)):**
        *   **Action (Scene Features)**: Modify [`extract_image_features`](cortexia_video/clip_wrapper.py:125) to accept `images_data: List[Image.Image]`.
            *   Preprocess each image using `self.image_preprocess`.
            *   Stack the resulting tensors into a batch (e.g., `torch.stack(processed_images_list)`).
            *   Pass this batch to `self.model.encode_image()`.
            *   Return type: `torch.Tensor` of shape `(batch_size, feature_dim)`.
        *   **In `ProcessingManager` (Scene Features)**: Call with the batch of PIL images. Assign results to `FrameData.scene_clip_features` for each item in the batch.
        *   **Action (Object Features)**: The same `extract_image_features` method will be used.
        *   **In `ProcessingManager` (Object Features)**: After segmentation, collect all *cropped object images* (as PIL Images) from all detections across the entire batch into a single flat list. Pass this list to `extract_image_features`. Then, carefully map the resulting batch of object features back to their respective `DetectionResult.object_clip_features` fields.

    *   **`ObjectDescriber.describe_object`**:
        *   No batching changes planned for this component. It will be called within a loop for individual segments if the "describe" mode is active.

5.  **Integrate Batched Processing in `ProcessingManager`:**
    *   The main loop in `process_video` will now manage passing entire batches of images/data to the refactored methods of `lister`, `detector`, `segmenter`, and `feature_extractor`.
    *   Crucially, after each batched call, implement logic to correctly map the batched results (which will be lists of results) back to the individual `FrameData` objects and their nested `DetectionResult`/`SegmentationResult` children within the current batch.

## Phase 2: Data Aggregation, Decoupling, and Logging (Largely Unchanged from Current)

6.  **Data Aggregation:**
    *   After all processing steps for a batch are complete, and the `FrameData` objects for that batch are fully populated, add them to the main `video_content_instance.frames` dictionary.

7.  **Decoupling Metadata and Image Data:**
    *   The existing Pydantic models ([`FrameData`](cortexia_video/schemes.py:68), [`DetectionResult`](cortexia_video/schemes.py:19), [`SegmentationResult`](cortexia_video/schemes.py:47)) and the [`save_annotations`](cortexia_video/data_io.py:44) function in `data_io.py` already handle the separation of JSON metadata from NumPy array data (masks, etc., saved via pickle). This core logic remains valid.

8.  **Maintaining JSON Logging:**
    *   The structure of the output JSON metadata file generated by [`save_annotations`](cortexia_video/data_io.py:44) should remain consistent with the current format due to the Pydantic models' `.dict()` methods.

## Mermaid Diagram of Proposed Workflow

```mermaid
graph TD
    A[Start process_video] --> B(Load Config: batch_size, frame_interval);
    B --> C[Initialize VideoReader];
    C --> D[Loop Over Batches from data_io.load_video_frames_batched];
    D -- Batch of (frame_meta_list, frame_np_list) --> E[Create List of FrameData objects for Batch (populate frame_num, timestamp, rgb_image)];
    E --> E_pil[Convert Batch rgb_images to PIL Images (or use np_arrays directly if components support)];

    subgraph Batch Processing Steps
        E_pil -- Batch of Images/NP Arrays --> F[lister.list_objects_in_image_batched(...)];
        F -- Batch of Lister Results --> F_prompts[Generate Batch of dino_prompts];
        F_prompts -- Batch of Prompts --> G_input;
        E_pil -- Batch of Images --> G_input;
        G_input --> G[detector.detect_objects_batched(...)];
        
        G -- Batch of Detection Lists --> H_prep_seg[Prepare Batch of Boxes for Segmenter];
        E_pil -- Batch of Images --> H_prep_seg;
        H_prep_seg --> H[segmenter.segment_object_batched(...)];

        E_pil -- Batch of Images --> I_scene[feature_extractor.extract_image_features_batched (Scene)];
        
        H -- Batch of Masks & Detections --> J_prep_obj_feat[Prepare Batch of Cropped Objects for Feature Extractor];
        J_prep_obj_feat --> J_obj[feature_extractor.extract_image_features_batched (Object)];

        I_scene -- Batch Scene Feats --> K_map;
        J_obj -- Batch Obj Feats --> K_map;
        H -- Batch Masks --> K_map;
        G -- Batch Detections --> K_map;
        F -- Batch Lister Results --> K_map;
        K_map[Map all batched results back to individual FrameData and DetectionResult objects in the current batch list];
    end

    K_map --> L_loop_describe[For each FrameData in Batch List (if "describe" mode active)];
        L_loop_describe -- FrameData & its Segments --> L_desc{Describer (Per Segment - No Batching)};
        L_desc --> L_desc1[describer.describe_object(...)];
        L_desc1 --> L_desc2[Update Detection with Description];
    L_desc2 --> L_loop_describe;
    L_loop_describe -- All Describe steps done for batch --> M_agg;
    
    M_agg[Add processed FrameData list to video_content_instance.frames];
    M_agg --> D;

    D -- All Batches Processed --> R[data_io.save_annotations(video_content_instance)];
    R --> S[End];