import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np # Added for type hinting if used

# Placeholder for future classes and functions
# e.g., SAM2ImageSegmentor, SAM2VideoSegmentor

class SAM2ImageSegmentor:
    def __init__(self, model_cfg: str, checkpoint: str):
        """
        Initializes the SAM2ImageSegmentor.

        Args:
            model_cfg (str): Path to the SAM2 model configuration file.
            checkpoint (str): Path to the SAM2 model checkpoint file.
        """
        sam_model = build_sam2(model_cfg, checkpoint)
        self.predictor = SAM2ImagePredictor(sam_model)

    def segment_image(self, image_np: np.ndarray, input_prompts: dict):
        """
        Segments an image using SAM2 based on input prompts.

        Args:
            image_np (np.ndarray): The input image as a NumPy array (HWC, RGB, uint8).
            input_prompts (dict): A dictionary containing prompts for the model.
                Expected keys:
                - "points" (optional): A torch.Tensor of shape (N, 2) or (B, N, 2) for point prompts.
                - "point_labels" (optional): A torch.Tensor of shape (N,) or (B, N) for point labels (1 for foreground, 0 for background).
                - "box" (optional): A torch.Tensor of shape (4,) or (B, 4) for box prompts (XYXY format).
                - "multimask_output" (optional): Bool, whether to return multiple masks. Defaults to True.

        Returns:
            Tuple: (masks, scores, logits)
                - masks (torch.Tensor): The output segmentation masks.
                - scores (torch.Tensor): The IOU predictions for each mask.
                - logits (torch.Tensor): The low-resolution mask logits.
        """
        # Convert NumPy image to PyTorch tensor
        # SAM2ImagePredictor.set_image expects an RGB image, uint8, HWC as a torch tensor.
        # Input image_np is assumed to be HWC, RGB, uint8.
        if not isinstance(image_np, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if image_np.dtype != np.uint8:
            # SAM2 expects uint8 images. If not, try to convert, but warn.
            # This might indicate an issue upstream or a need for more robust preprocessing.
            print(f"Warning: Input image dtype is {image_np.dtype}, converting to uint8. SAM2 expects uint8.")
            if image_np.max() <= 1.0 and image_np.min() >=0.0 : # Assuming float image in [0,1]
                image_np = (image_np * 255).astype(np.uint8)
            else: # Other cases might need specific handling or will raise errors in from_numpy
                image_np = image_np.astype(np.uint8)


        if image_np.ndim == 2:  # Grayscale
            print("Info: Input image is grayscale. Converting to RGB by repeating channels.")
            image_rgb_np = np.stack((image_np,) * 3, axis=-1)
        elif image_np.shape[2] == 1:  # Grayscale with channel dim
            print("Info: Input image is grayscale (H, W, 1). Converting to RGB by repeating channels.")
            image_rgb_np = np.concatenate([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:  # RGBA
            print("Info: Input image is RGBA. Converting to RGB by removing alpha channel.")
            image_rgb_np = image_np[..., :3]
        elif image_np.shape[2] == 3: # RGB or BGR
            image_rgb_np = image_np # Assume it's RGB as per SAM2 docs
        else:
            raise ValueError(f"Unsupported number of image channels: {image_np.shape[2]}. Expected 1 (grayscale), 3 (RGB), or 4 (RGBA).")

        image_tensor = torch.from_numpy(image_rgb_np) # HWC, RGB, uint8

        # Ensure the image tensor is on the same device as the model
        device = next(self.predictor.model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Set the image in the predictor
        # SAM2ImagePredictor.set_image also handles moving to device if not already
        # and permuting to CHW if model expects that (though SAM2 seems to use HWC internally for set_image)
        self.predictor.set_image(image_tensor)

        # Prepare predictor arguments from input_prompts
        # SAM2ImagePredictor.predict API:
        # points (Optional[torch.Tensor])
        # point_labels (Optional[torch.Tensor])
        # box (Optional[torch.Tensor])
        # mask_input (Optional[torch.Tensor])
        # multimask_output (bool = True)
        # random_color_setting (Optional[Any] = None)
        # box_points_per_side (Optional[int] = None)
        # Return: masks, iou_predictions, low_res_masks

        predict_kwargs = {
            "points": input_prompts.get("points"),
            "point_labels": input_prompts.get("point_labels"),
            "box": input_prompts.get("box"),
            "mask_input": input_prompts.get("mask_input"),
            "multimask_output": input_prompts.get("multimask_output", True),
        }
        # Filter out None values, as predictor expects either a Tensor or None, not {"box": None}
        predict_kwargs = {k: v for k, v in predict_kwargs.items() if v is not None}


        # Perform prediction
        # Use autocast for mixed precision inference if on CUDA
        autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
        # bfloat16 might not be supported on all CPUs or older GPUs.
        # For broader compatibility, float16 could be an alternative on CUDA.
        # SAM2 examples often use bfloat16 for CUDA.
        # If device is CPU, torch.autocast for "cpu" uses bfloat16 if available, otherwise float32.
        dtype = torch.bfloat16 if autocast_device_type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

        with torch.inference_mode(), torch.autocast(autocast_device_type, dtype=dtype):
            masks, scores, logits = self.predictor.predict(**predict_kwargs)

        return masks, scores, logits

# SAM2VideoSegmentor will be added later

class SAM2VideoSegmentor:
    def __init__(self, model_cfg: str, checkpoint: str):
        """
        Initializes the SAM2VideoSegmentor.

        Args:
            model_cfg (str): Path to the SAM2 model configuration file.
            checkpoint (str): Path to the SAM2 model checkpoint file.
        """
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        self.device = next(self.predictor.model.parameters()).device
        # Determine autocast dtype once during initialization for consistency
        self.autocast_dtype = torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32


    def _prepare_single_frame(self, frame_np: np.ndarray) -> torch.Tensor:
        """
        Prepares a single video frame:
        - Validates dimensions.
        - Converts to uint8 if necessary.
        - Converts grayscale or RGBA to RGB (HWC).
        - Converts NumPy array to PyTorch tensor on the model's device.

        Args:
            frame_np (np.ndarray): A single frame as a NumPy array (HWC or HW).

        Returns:
            torch.Tensor: The processed frame as a PyTorch tensor (HWC, RGB, uint8) on the model's device.
        """
        if not isinstance(frame_np, np.ndarray):
            raise TypeError("Input frame must be a NumPy array.")

        if frame_np.ndim not in [2, 3]:
            raise ValueError(f"Frame must be 2D (HW) or 3D (HWC), got {frame_np.ndim} dimensions.")

        if frame_np.dtype != np.uint8:
            print(f"Warning: Frame dtype is {frame_np.dtype}, converting to uint8. SAM2 expects uint8.")
            if frame_np.max() <= 1.0 and frame_np.min() >= 0.0:  # Assuming float image in [0,1]
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

        processed_frame_np = frame_np
        if frame_np.ndim == 2:  # Grayscale HW
            print("Info: Input frame is grayscale (HW). Converting to RGB by repeating channels.")
            processed_frame_np = np.stack((frame_np,) * 3, axis=-1)
        elif frame_np.shape[2] == 1:  # Grayscale HWC with C=1
            print("Info: Input frame is grayscale (HWC with C=1). Converting to RGB by repeating channels.")
            processed_frame_np = np.concatenate([frame_np] * 3, axis=-1)
        elif frame_np.shape[2] == 4:  # RGBA HWC with C=4
            print("Info: Input frame is RGBA. Converting to RGB by removing alpha channel.")
            processed_frame_np = frame_np[..., :3]

        if processed_frame_np.shape[2] != 3:
            raise ValueError(f"Processed frame must have 3 channels (RGB), got {processed_frame_np.shape[2]}. Original shape {frame_np.shape}")

        return torch.from_numpy(processed_frame_np).to(self.device)

    def _prepare_frames(self, video_frames_np) -> list[torch.Tensor]:
        """
        Prepares video frames for the predictor.
        Accepts either a list of NumPy arrays (frames) or a single THWC NumPy array.

        Args:
            video_frames_np (list[np.ndarray] | np.ndarray): Video frames.

        Returns:
            list[torch.Tensor]: A list of processed frames (PyTorch tensors on model device).
        """
        processed_frames = []
        if isinstance(video_frames_np, np.ndarray) and video_frames_np.ndim == 4: # THWC
            for i in range(video_frames_np.shape[0]):
                frame = video_frames_np[i]
                processed_frames.append(self._prepare_single_frame(frame))
        elif isinstance(video_frames_np, list):
            for frame_np in video_frames_np:
                processed_frames.append(self._prepare_single_frame(frame_np))
        else:
            raise TypeError("video_frames_np must be a list of NumPy arrays or a single THWC NumPy array.")

        return processed_frames

    def init_video_state(self, video_frames_np):
        """
        Initializes the video state for tracking using the provided frames.

        Args:
            video_frames_np (list[np.ndarray] | np.ndarray): The video frames.

        Returns:
            The initialized state object from the SAM2 video predictor.
        """
        processed_frames = self._prepare_frames(video_frames_np)

        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.autocast_dtype):
            # SAM2 Video Predictor's init_state expects a list of HWC, RGB, uint8 tensors
            state = self.predictor.init_state(processed_frames)
        return state

    def add_prompts_and_get_masks(self, state, frame_idx: int, object_ids, input_prompts: dict):
        """
        Adds new prompts (points or boxes) for specified object IDs at a given frame index
        and gets the resulting segmentation masks.

        Args:
            state: The current video state from init_video_state or propagation.
            frame_idx (int): The index of the frame to add prompts to.
            object_ids (list[int] | torch.Tensor): IDs of the objects being prompted.
            input_prompts (dict): Dictionary with prompt information.
                Expected keys:
                - "points" (optional): Point coordinates, e.g., list of [x,y] or Tensor.
                - "point_labels" (optional): Labels for points (1 for fg, 0 for bg).
                - "box" (optional): Box coordinates (XYXY), e.g., list [x1,y1,x2,y2] or Tensor.

        Returns:
            Tuple: (frame_idx_out, obj_ids_out, masks) from the predictor.
        """
        points_val = input_prompts.get("points")
        labels_val = input_prompts.get("point_labels")
        box_val = input_prompts.get("box")

        points_t = torch.as_tensor(points_val, device=self.device, dtype=torch.float) if points_val is not None else None
        labels_t = torch.as_tensor(labels_val, device=self.device, dtype=torch.int) if labels_val is not None else None
        box_t = torch.as_tensor(box_val, device=self.device, dtype=torch.float) if box_val is not None else None

        if isinstance(object_ids, list):
            obj_ids_t = torch.as_tensor(object_ids, device=self.device, dtype=torch.long)
        elif isinstance(object_ids, torch.Tensor):
            obj_ids_t = object_ids.to(device=self.device, dtype=torch.long)
        else:
            raise TypeError("object_ids must be a list or a torch.Tensor.")

        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.autocast_dtype):
            # API: add_new_points_or_box(state, frame_idx, obj_ids, points=None, point_labels=None, box=None)
            frame_idx_out, obj_ids_out, masks = self.predictor.add_new_points_or_box(
                state,
                frame_idx,
                obj_ids_t,
                points=points_t,
                point_labels=labels_t,
                box=box_t
            )
        return frame_idx_out, obj_ids_out, masks

    def propagate_masks_in_video(self, state):
        """
        Propagates masks throughout the video sequence based on the current state.
        This is a generator that yields results frame by frame.

        Args:
            state: The current video state.

        Yields:
            Tuple: (frame_idx, obj_ids, masks) for each frame processed during propagation.
        """
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.autocast_dtype):
            for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(state):
                yield frame_idx, obj_ids, masks
