
# %% [markdown]
# # Use Cortexia with Lance Dataset
# 
# Example: Use Cortexia with a Lance table.
# 
# This example shows how to:
# - Read images stored as bytes from a Lance dataset column
# - Run features: Caption, Listing
# - Use Listing tags as prompts for Detection, then run Segmentation
# - Save annotated results to a new Lance table (or Parquet fallback)
# 
# Assumptions:
# - The Lance dataset at `dummys/lance_data/all_in_one.lance` exists.
# - Image bytes are stored in the `camera_left` column (e.g., JPEG/PNG bytes).
# - The table has no annotations; other columns (video/frame ids) are optional.
# 
# If your columns differ, set the env vars or change the defaults below.

# %% [markdown]
# ## Setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# Import deps 
from __future__ import annotations

import os
import sys
import io
import json
import lance
import math
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image

import pyarrow as pa


parent_path = str(Path.cwd().parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)
REPO_ROOT = parent_path

import cortexia
from cortexia.data.models.video import VideoFramePacket
from cortexia.data.io.batch_processor import BatchProcessor
from cortexia.data.models.video import TrajectoryPoint
from cortexia.data.models.result.caption_result import CaptionResult
from cortexia.data.models.result.detection_result import DetectionResult

# %%
# os.environ["HF_HOME"]="/vita-vepfs-data/fileset1/model/heng.li/huggingface"

# %%
# Make local package importable when running from cookbook/
# parent_path = str(Path.cwd().parent)
# if parent_path not in sys.path:
#     sys.path.append(parent_path)
# REPO_ROOT = parent_path

# %%
# import cotexia related thing 
# import cortexia
# from cortexia.data.models.video import VideoFramePacket

# %%

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
_repo_path = Path(REPO_ROOT)  # convert str to Path for safe joining

DATASET_PATH = os.environ.get(
    "LANCE_DATASET",
    None
)
if not DATASET_PATH:
    raise ValueError("LANCE_DATASET env var must be set")

# Column names (customize as needed)
IMAGE_COL = os.environ.get("LANCE_IMAGE_COL", "camera_left")
VIDEO_ID_COL = os.environ.get("LANCE_VIDEO_ID_COL", "trigger")   # e.g., "video_id" if present
FRAME_NUM_COL = os.environ.get("LANCE_FRAME_NUM_COL", None) # e.g., "frame_number" if present
TIMESTAMP_COL = os.environ.get("LANCE_TIMESTAMP_COL", "timestamp") # optional ms/seconds; default to index/30

CAPTION_COL = "cortexia_caption"
LISTING_COL = "cortexia_tags"
DETECTION_COL = "cortexia_detection"
SEGMENTATION_COL = "cortexia_segmentation"
TRAJECTORY_COL = "cortexia_trajectory"

TRAJECTORY_LENGTH = 7

# Output path for annotated table
OUTPUT_LANCE = os.environ.get(
    "LANCE_OUTPUT",
    DATASET_PATH
)
if not OUTPUT_LANCE:
    raise ValueError("If set LANCE_OUTPUT env var, it cannot be set empty")

# Limit rows for demo (e.g., 8). Use 0 or unset to disable limiting.
ROW_LIMIT = int(os.environ.get("LANCE_ROW_LIMIT", 0))

# Write mode: "create" (default) creates a new table at OUTPUT_LANCE.
# Set LANCE_WRITE_MODE="update" to merge results back into DATASET_PATH.
LANCE_WRITE_MODE = os.environ.get("LANCE_WRITE_MODE", "create").lower()



# %% [markdown]
# ## Helpers to work with lance

# %%

def load_lance_table(dataset_path: str) -> pa.Table:
    """Load the entire Lance dataset into a PyArrow Table.

    For simplicity of the cookbook example we load all rows. For large datasets,
    adapt to stream batches or filter rows.
    """
    # try:
    #     import lance
    # except Exception as e:
    #     raise RuntimeError(
    #         "Lance Python package is required for this example. Install 'pylance'."
    #     ) from e

    ds = lance.dataset(dataset_path)
    # Convert to Arrow table (small demo dataset assumed)
    table = ds.to_table()
    # 检查并转换binary类型列为large_binary
    modified_table = table
    for col_name in table.column_names:
        col = table.column(col_name)
        if pa.types.is_binary(col.type):
            # 将binary转换为large_binary
            new_col = col.cast(pa.large_binary())
            # 替换列（使用已有的表创建新表）
            idx = modified_table.schema.get_field_index(col_name)
            modified_table = modified_table.set_column(idx, col_name, new_col)
    del ds
    del table
    return modified_table

def create_streaming_loader(dataset_path: str, batch_size: int = 32):
    """创建一个流式加载器，而不是一次性加载整个数据集"""
    ds = lance.dataset(dataset_path)
    schema = ds.schema
    
    # 检查并转换binary类型列为large_binary
    # for col_name in schema.names:
    #     col = schema.field(col_name)
    #     if pa.types.is_binary(col.type):
    #         # 转换schema但不实际加载数据
    #         schema = schema.set(col_name, pa.field(col_name, pa.large_binary()))
    
    def stream_batches():
        """流式生成数据批次"""
        for batch in ds.to_batches(batch_size=batch_size):
            # 按需转换binary列为large_binary
            modified_batch = batch
            for col_name in batch.column_names:
                col = batch.column(col_name)
                if pa.types.is_binary(col.type):
                    new_col = col.cast(pa.large_binary())
                    idx = modified_batch.schema.get_field_index(col_name)
                    modified_batch = modified_batch.set_column(idx, col_name, new_col)
            yield modified_batch
    
    return stream_batches, schema

def decode_image_from_bytes(b: bytes) -> np.ndarray:
    """Decode image bytes (e.g., JPEG/PNG) into an RGB numpy array."""
    with Image.open(io.BytesIO(b)) as im:
        im = im.convert("RGB")
        return np.array(im)


def build_video_frame_packet(row: pa.Table, row_idx: int, next_table: Optional[pa.Table] = None) -> VideoFramePacket:
    """Construct a VideoFramePacket from a 1-row Arrow table slice."""
    # Image
    img_val = row[IMAGE_COL][0]
    if hasattr(img_val, "as_py"):
        img_val = img_val.as_py()
    frame_np = decode_image_from_bytes(img_val)

    # Video/frame/timestamp
    if VIDEO_ID_COL and VIDEO_ID_COL in row.column_names:
        vid = str(row[VIDEO_ID_COL][0])
    else:
        vid = "lance_demo"

    if FRAME_NUM_COL and FRAME_NUM_COL in row.column_names:
        frame_no = int(row[FRAME_NUM_COL][0])
    else:
        frame_no = int(row_idx)

    if TIMESTAMP_COL and TIMESTAMP_COL in row.column_names:
        ts_val = row[TIMESTAMP_COL][0]
        if hasattr(ts_val, "as_py"):
            ts_val = ts_val.as_py()
        # Interpret as seconds if float, ms if int
        if isinstance(ts_val, float):
            ts = datetime.timedelta(seconds=ts_val/1e9)
        elif isinstance(ts_val, int):
            ts = datetime.timedelta(milliseconds=ts_val/1e6)
        else:
            ts = datetime.timedelta(seconds=frame_no / 30.0)
    else:
        ts = datetime.timedelta(seconds=frame_no / 30.0)

    additional_metadata = {}
    if LISTING_COL in row.column_names:
        listing = row[LISTING_COL][0]
        if hasattr(listing, "as_py"):
            listing = listing.as_py()
        additional_metadata["lister_results"] = list(listing["tags"]) if "tags" in listing else []
    if DETECTION_COL in row.column_names:
        detection = row[DETECTION_COL][0]
        if hasattr(detection, "as_py"):
            detection = detection.as_py()
        additional_metadata["detections"] = DetectionResult.from_dict(detection)

    # Build trajectory data using current + next 6 frames (7 points total)
    trajectory_points = []
    if next_table is not None:
        # Import trajectory models
        # from cortexia.data.models.video import TrajectoryPoint
        # Collect trajectory points for current frame and next 6 frames
        for i, data in enumerate(next_table):
            if "odo" in data:
                odo_data = data['odo'][0]
                if hasattr(odo_data, 'as_py'):
                    odo_data = odo_data.as_py()
                
                # odo_data should contain 7 numbers: x, y, z, qx, qy, qz, qw
                if isinstance(odo_data, (list, tuple)) and len(odo_data) >= 7:
                    x, y, z, qx, qy, qz, qw = odo_data[:7]
                    traj_point = TrajectoryPoint(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)
                else:
                    # Create default trajectory point if invalid odo data
                    traj_point = TrajectoryPoint(x=(row_idx+i+1)*0.1, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
            else:
                # No odo column, create simulated trajectory data
                traj_point = TrajectoryPoint(x=(row_idx+i+1)*0.1, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
            
            trajectory_points.append(traj_point)

    return VideoFramePacket(
        frame_data=frame_np,
        frame_number=frame_no,
        timestamp=ts,
        source_video_id=vid,
        additional_metadata=additional_metadata,
        trajectory=trajectory_points,  # Note: field name has typo
        current_traj_index=0  # Current frame is always at index 0
    )


def make_loader(table: pa.Table):
    """Create a BatchProcessor-compatible loader over a fixed Arrow table."""
    def load_func(indices: List[int]) -> List[VideoFramePacket]:
        # Take a subset table by row indices and convert to packets
        sub = table.take(pa.array(indices))
        frames: List[VideoFramePacket] = []
        for pos, row_idx in enumerate(indices):
            one = sub.slice(pos, 1)
            frames.append(build_video_frame_packet(one, row_idx, full_table=table))
        return frames

    return load_func

def results_to_struct_array(results):
    if not results:
        return pa.array([], type=pa.null())
    first_struct = results[0].to_pyarrow_struct()
    struct_type = first_struct.type
    names = struct_type.names
    dicts = []
    for r in results:
        s = r.to_pyarrow_struct()
        scalar = s[0]
        row = {}
        for name in names:
            try:
                val = scalar[name]
                row[name] = val.as_py() if val is not None else None
            except KeyError:
                row[name] = None
        dicts.append(row)
    return pa.array(dicts, type=struct_type)


def limit_table(table: pa.Table, limit: int | None) -> pa.Table:
    """Return a slice of the table limited to `limit` rows (if > 0)."""
    try:
        if limit is None or int(limit) <= 0:
            return table
        limit = min(int(limit), len(table))
        return table.slice(0, limit)
    except Exception:
        return table
    

def write_table(dataset_path: str, col: str, infos: pa.array, write_mode: str = "create", output_path: str = None):
    """
    Write results to a Lance dataset.
    
    Args:
        dataset_path: Path to the input dataset
        col: Column name to add/replace
        infos: PyArrow array with the data to write
        write_mode: "create" (default) or "update"
        output_path: Output path for "create" mode (defaults to dataset_path for "update" mode)
    """
    if write_mode == "update":
        # Merge results into the existing Lance dataset
        ds = lance.dataset(dataset_path)
        if ds.count_rows() != len(infos):
            raise ValueError("infos must have the same number of rows as the dataset")
        
        table = ds.to_table()
        
        # Check if column already exists
        if col in table.column_names:
            # Replace existing column
            idx = table.column_names.index(col)
            table = table.set_column(idx, col, infos)
        else:
            # Add new column
            table = table.append_column(col, infos)
        
        lance.write_dataset(table, dataset_path, mode="overwrite")
        print(f"Results merged into existing Lance dataset: {dataset_path}")
    else:
        # Write to a new Lance dataset (default "create" mode)
        if output_path is None:
            raise ValueError("output_path must be specified for 'create' mode")
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Overwrite destination if exists by writing a fresh dataset
        if output_path_obj.exists():
            # Best-effort cleanup; Lance manages versions but this is a demo
            import shutil
            shutil.rmtree(output_path_obj, ignore_errors=True)
        
        # For "create" mode, we need to load the original table and add the new column
        ds = lance.dataset(dataset_path)
        table = ds.to_table()
        
        # Add the new column
        table = table.append_column(col, infos)
        
        lance.write_dataset(table, str(output_path_obj))
        print(f"Results written to new Lance dataset: {output_path}")

def get_features_map():
    return {
        "caption": CAPTION_COL,
        "listing": LISTING_COL,
        "detection": DETECTION_COL,
        "segmentation": SEGMENTATION_COL,
        "trajectory": TRAJECTORY_COL,
    }

class FeatureProcessor:
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.feature = None
        
    def __enter__(self):
        self.feature = cortexia.create_feature(self.feature_name)
        return self.feature
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.feature, '_release'):
            self.feature._release()
        self.feature = None

def run():
    # Import Path for output path handling
    from pathlib import Path
    
    features = get_features_map()
    for feature, col in features.items():
        print(f"Running feature: {feature}")
        output_batches = []
        with FeatureProcessor(feature) as feature:
            print("Loading Lance dataset for streaming processing:", DATASET_PATH)
            stream_batches, schema = create_streaming_loader(DATASET_PATH, batch_size=512)
            row_idx = 0
            output_batches = []
            for batch_idx, batch in enumerate(stream_batches()):
                length, frames = len(batch), []
                for idx in range(batch.num_rows):
                    one_row = batch.slice(idx, 1)
                    if idx < length - 1:
                        next_table = batch.slice(idx+1, min(TRAJECTORY_LENGTH, length-idx-1))
                    else:
                        next_table = None
                    packet = build_video_frame_packet(one_row, row_idx, next_table)
                    frames.append(packet)
                    row_idx += 1
                if len(frames) > 0:
                    results = feature.process_batch(frames)
                    if results is not None:
                        output_batches.extend(results)
        if len(output_batches) > 0:
            output_table = results_to_struct_array(output_batches)
            if LANCE_WRITE_MODE == "update":
                write_table(DATASET_PATH, col, output_table, write_mode="update")
            else:
                # For "create" mode, we need to define an output path
                # Using the same OUTPUT_LANCE variable as in use_with_lance.py for consistency
                output_path = os.environ.get(
                    "LANCE_OUTPUT",
                    str(Path(REPO_ROOT) / "dummys" / "lance_data" / "all_in_one_stream_annotated.lance")
                )
                write_table(DATASET_PATH, col, output_table, write_mode="create", output_path=output_path)
            # output_table = output_table.append_column(pa.field(feature, output_table.type))

run()      
