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
%load_ext autoreload
%autoreload 2

# %%
# Import deps 
from __future__ import annotations

import os
import sys
import io
import json
import math
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image

import pyarrow as pa


# %%
os.environ["HF_HOME"]="/vita-vepfs-data/fileset1/model/heng.li/huggingface"

# %%
# Make local package importable when running from cookbook/
parent_path = str(Path.cwd().parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)
REPO_ROOT = parent_path

# %%
# import cotexia related thing 
import cortexia
from cortexia.data.models.video import VideoFramePacket

# %%

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
_repo_path = Path(REPO_ROOT)  # convert str to Path for safe joining

DATASET_PATH = os.environ.get(
    "LANCE_DATASET",
    str(_repo_path / "dummys" / "lance_data" / "all_in_one.lance")
)

# Column names (customize as needed)
IMAGE_COL = os.environ.get("LANCE_IMAGE_COL", "camera_left")
VIDEO_ID_COL = os.environ.get("LANCE_VIDEO_ID_COL", None)   # e.g., "video_id" if present
FRAME_NUM_COL = os.environ.get("LANCE_FRAME_NUM_COL", None) # e.g., "frame_number" if present
TIMESTAMP_COL = os.environ.get("LANCE_TIMESTAMP_COL", None) # optional ms/seconds; default to index/30

# Output path for annotated table
OUTPUT_LANCE = os.environ.get(
    "LANCE_OUTPUT",
    str(_repo_path / "dummys" / "lance_data" / "all_in_one_annotated.lance")
)
OUTPUT_PARQUET = os.environ.get(
    "PARQUET_OUTPUT",
    str(_repo_path / "dummys" / "lance_data" / "all_in_one_annotated.parquet")
)
# Limit rows for demo (e.g., 8). Use 0 or unset to disable limiting.
ROW_LIMIT = int(os.environ.get("LANCE_ROW_LIMIT", "8"))



# %% [markdown]
# ## Helpers to work with lance

# %%

def load_lance_table(dataset_path: str) -> pa.Table:
    """Load the entire Lance dataset into a PyArrow Table.

    For simplicity of the cookbook example we load all rows. For large datasets,
    adapt to stream batches or filter rows.
    """
    try:
        import lance
    except Exception as e:
        raise RuntimeError(
            "Lance Python package is required for this example. Install 'lance'."
        ) from e

    ds = lance.dataset(dataset_path)
    # Convert to Arrow table (small demo dataset assumed)
    tbl = ds.to_table()
    return tbl


def decode_image_from_bytes(b: bytes) -> np.ndarray:
    """Decode image bytes (e.g., JPEG/PNG) into an RGB numpy array."""
    with Image.open(io.BytesIO(b)) as im:
        im = im.convert("RGB")
        return np.array(im)


def build_video_frame_packet(row: pa.Table, row_idx: int, full_table: Optional[pa.Table] = None) -> VideoFramePacket:
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
            ts = datetime.timedelta(seconds=ts_val)
        elif isinstance(ts_val, int):
            ts = datetime.timedelta(milliseconds=ts_val)
        else:
            ts = datetime.timedelta(seconds=frame_no / 30.0)
    else:
        ts = datetime.timedelta(seconds=frame_no / 30.0)

    # Build trajectory data using current + next 6 frames (7 points total)
    trajectory_points = []
    if full_table is not None:
        # Import trajectory models
        from cortexia.data.models.video import TrajectoryPoint
        
        # Collect trajectory points for current frame and next 6 frames
        for j in range(7):  # 0 to 6
            frame_idx = row_idx + j
            if frame_idx >= len(full_table):
                # Not enough future frames, break early
                break
                
            # Get odo data for this frame
            if 'odo' in full_table.column_names:
                odo_data = full_table['odo'][frame_idx]
                if hasattr(odo_data, 'as_py'):
                    odo_data = odo_data.as_py()
                
                # odo_data should contain 7 numbers: x, y, z, qx, qy, qz, qw
                if isinstance(odo_data, (list, tuple)) and len(odo_data) >= 7:
                    x, y, z, qx, qy, qz, qw = odo_data[:7]
                    traj_point = TrajectoryPoint(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)
                else:
                    # Create default trajectory point if invalid odo data
                    traj_point = TrajectoryPoint(x=frame_idx*0.1, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
            else:
                # No odo column, create simulated trajectory data
                traj_point = TrajectoryPoint(x=frame_idx*0.1, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
            
            trajectory_points.append(traj_point)

    return VideoFramePacket(
        frame_data=frame_np,
        frame_number=frame_no,
        timestamp=ts,
        source_video_id=vid,
        additional_metadata={},
        trajecotry=trajectory_points,  # Note: field name has typo
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

# %% [markdown]
# ## Create Features and get Lance Tables

# %%
print("Loading Lance table:", DATASET_PATH)
table = load_lance_table(DATASET_PATH)
# We Limit Table for demo runs
table = limit_table(table, ROW_LIMIT)
schema_cols = set(table.column_names)
if IMAGE_COL not in schema_cols:
    raise ValueError(f"Column '{IMAGE_COL}' not found. Available: {sorted(schema_cols)}")

n_rows = len(table)
print(f"Rows: {n_rows}; image column: '{IMAGE_COL}'")

# %%
cortexia.list_features()

# %%
# Indices are row numbers
indices = list(range(n_rows))
load_func = make_loader(table)

# Features
caption = cortexia.create_feature("caption")
listing = cortexia.create_feature("listing")
detection = cortexia.create_feature("detection")
segmentation = cortexia.create_feature("segmentation")
trajectory = cortexia.create_feature("trajectory")

# Materialize frames once for chaining and attachment
frames = load_func(indices)

# %% [markdown]
# **Why do we use add_annotation_result** 

# %%
from cortexia.data.io.batch_processor import BatchProcessor
print("Running Caption via BatchProcessor and attaching to frames...")
frames_map = {idx: frames[idx] for idx in indices}
bp = BatchProcessor(batch_size=4)
bp.load_indices(indices)

def bp_load(batch_indices: List[int]) -> List[VideoFramePacket]:
    return [frames_map[i] for i in batch_indices]

def bp_infer(fr_batch: List[VideoFramePacket], batch_indices: List[int]):
    return caption.process_batch(fr_batch)

def bp_save(idx: int, result):
    frames_map[idx].add_annotation_result(result)

# Process in batches and attach directly using save_func
_ = bp.process_batch(load_func=bp_load, inference_func=bp_infer, save_func=bp_save, filter_func=None)

# Collect attached caption results back from frames for later writing
cap_results = []
for f in frames:
    if f.annotations and 'CaptionResult' in f.annotations:
        cap_results.append(f.annotations['CaptionResult'])
    else:
        from cortexia.data.models.result.caption_result import CaptionResult
        cap_results.append(CaptionResult(caption=""))


# %%
# 2) Listing (BatchProcessor chain; attach + set prompts for detection)
print("Running Listing via BatchProcessor and attaching to frames...")
list_results_map = {}
bp2 = BatchProcessor(batch_size=4)
bp2.load_indices(indices)

def bp2_load(batch_indices: List[int]) -> List[VideoFramePacket]:
    return [frames_map[i] for i in batch_indices]

def bp2_infer(fr_batch: List[VideoFramePacket], batch_indices: List[int]):
    return listing.process_batch(fr_batch)

def bp2_save(idx: int, result):
    f = frames_map[idx]
    f.add_annotation_result(result)
    # Also provide prompts for detection via metadata
    f.additional_metadata["lister_results"] = list(getattr(result, 'tags', []) or [])
    list_results_map[idx] = result

_ = bp2.process_batch(load_func=bp2_load, inference_func=bp2_infer, save_func=bp2_save, filter_func=None)
list_results = [list_results_map[i] for i in indices]

# %%
list_results[0].tags

# %%
# 3) Detection
print("Running Detection (prompted by Listing tags) and attaching...")
det_results = detection.process_batch(frames)
for f, r in zip(frames, det_results):
    f.add_annotation_result(r)

# %%
det_results[0].detections

# %%
# 4) Segmentation
print("Running Segmentation (using Detection boxes) and attaching...")
seg_results = segmentation.process_batch(frames)

# %%
print(seg_results[0].segmentations[0])

# %% [markdown]
# **5) Trajectory Analysis - Annotate Action States**
# 
# Now let's use the trajectory feature to analyze movement states.
# The trajectory data is already loaded into VideoFramePacket from the 'odo' column 
# during frame creation, containing 7 numbers: x, y, z, qx, qy, qz, qw representing 
# position and orientation for current + next 6 frames (7 points total).

# %%
print("Running Trajectory Analysis to annotate action states...")

# Process trajectory analysis (trajectory data is already in frames)
traj_results = trajectory.process_batch(frames)

# Attach trajectory results to frames
for frame, traj_result in zip(frames, traj_results):
    frame.add_annotation_result(traj_result)

# %%
# Examine trajectory analysis results
print("Trajectory Analysis Results:")
for i in range(min(3, len(frames))):
    traj_result = traj_results[i]
    current_state = traj_result.get_current_state()
    state_dist = traj_result.state_distribution
    
    print(f"Frame {i}:")
    print(f"  Current state: {current_state}")
    print(f"  State distribution: {state_dist}")
    
    # Get current trajectory point details
    if 0 <= traj_result.current_index < len(traj_result.trajectory_points):
        point = traj_result.trajectory_points[traj_result.current_index]
        print(f"  Position: ({point.x:.2f}, {point.y:.2f}, {point.z:.2f})")
        print(f"  Orientation: yaw={math.degrees(point.yaw):.1f}°")
        print(f"  Velocity: {point.velocity:.3f}")
    print()

# %% [markdown]
# **Lets check some examples** 

# %%
for i in range(min(2, n_rows)):
    print(f"Row {i} -> caption: {cap_results[i].caption!r}")
    print(f"Row {i} -> tags: {list_results[i].tags}")
    # Handle new DetectionResult format with multiple detections
    if det_results[i].has_detections:
        # Show first detection as example
        first_det = det_results[i].detections[0]
        print(
            f"Row {i} -> det: count={det_results[i].count}, first: label={first_det.label!r}, score={first_det.score:.3f}, box={first_det.box.xyxy}"
        )
    else:
        print(f"Row {i} -> det: count=0, no detections")
    # Handle new SegmentationResult format with multiple segmentations
    if seg_results[i].has_segmentations:
        # Show first segmentation as example
        first_seg = seg_results[i].segmentations[0]
        print(
            f"Row {i} -> seg: count={seg_results[i].count}, first: label={first_seg.label!r}, area={first_seg.area}, mask.shape={first_seg.mask.shape}"
        )
    else:
        print(f"Row {i} -> seg: count=0, no segmentations")

# %% [markdown]
# **Lets write it into a table with annotation result**

# %%
col_caption_struct = results_to_struct_array(cap_results)
col_tags_struct = results_to_struct_array(list_results)
col_det_struct = results_to_struct_array(det_results)
col_seg_struct = results_to_struct_array(seg_results)
col_traj_struct = results_to_struct_array(traj_results)

annotated = table
annotated = annotated.append_column("cortexia_caption", col_caption_struct)
annotated = annotated.append_column("cortexia_tags", col_tags_struct)
annotated = annotated.append_column("cortexia_detection", col_det_struct)
annotated = annotated.append_column("cortexia_segmentation", col_seg_struct)
annotated = annotated.append_column("cortexia_trajectory", col_traj_struct)

# %%
try:
    import lance
    out = Path(OUTPUT_LANCE)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite destination if exists by writing a fresh dataset
    if out.exists():
        # Best-effort cleanup; Lance manages versions but this is a demo
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    lance.write_dataset(annotated, str(out))
    print(f"Annotated Lance dataset written to: {out}")
    wrote = True
except Exception as e:
    print(f"Lance write failed or unavailable: {e}")

# %%
# Take a quick preview for this table
def preview_rows(tbl: pa.Table, k: int = 3):
    print("Previewing first", min(k, len(tbl)), "rows (selected columns):")
    cols_to_show = [
        c for c in [
            VIDEO_ID_COL or None,
            FRAME_NUM_COL or None,
            "cortexia_caption",
            "cortexia_tags",
            "cortexia_detection",
            "cortexia_segmentation",
            "cortexia_trajectory",
        ] if c and c in tbl.column_names
    ]
    for i in range(min(k, len(tbl))):
        row = tbl.slice(i, 1)
        summary = {}
        for c in cols_to_show:
            cell = row[c][0]
            try:
                summary[c] = cell.as_py()
            except Exception:
                summary[c] = str(cell)
        print(f"Row {i}:", summary)

preview_rows(annotated, k=3)

# %%



