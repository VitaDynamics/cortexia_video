"""
Example: Use Cortexia with a Lance table.

This example shows how to:
- Read images stored as bytes from a Lance dataset column
- Run features: Caption, Listing
- Use Listing tags as prompts for Detection, then run Segmentation
- Save annotated results to a new Lance table (or Parquet fallback)

Assumptions:
- The Lance dataset at `dummys/lance_data/all_in_one.lance` exists.
- Image bytes are stored in the `camera_left` column (e.g., JPEG/PNG bytes).
- The table has no annotations; other columns (video/frame ids) are optional.

If your columns differ, set the env vars or change the defaults below.
"""

from __future__ import annotations

import os
import sys
import io
import json
import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image

import pyarrow as pa

# Make local package importable when running from cookbook/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import cortexia
from cortexia.api.cortexia import process_batch_with_processor
from cortexia.data.models.video import VideoFramePacket
from cortexia.data.io.batch_processor import BatchProcessor


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATASET_PATH = os.environ.get("LANCE_DATASET", str(REPO_ROOT / "dummys/lance_data/all_in_one.lance"))

# Column names (customize as needed)
IMAGE_COL = os.environ.get("LANCE_IMAGE_COL", "camera_left")
VIDEO_ID_COL = os.environ.get("LANCE_VIDEO_ID_COL", None)  # e.g., "video_id" if present
FRAME_NUM_COL = os.environ.get("LANCE_FRAME_NUM_COL", None)  # e.g., "frame_number" if present
TIMESTAMP_COL = os.environ.get("LANCE_TIMESTAMP_COL", None)  # optional ms/seconds; default to index/30

# Output path for annotated table
OUTPUT_LANCE = os.environ.get("LANCE_OUTPUT", str(REPO_ROOT / "dummys/lance_data/all_in_one_annotated.lance"))
OUTPUT_PARQUET = os.environ.get("PARQUET_OUTPUT", str(REPO_ROOT / "dummys/lance_data/all_in_one_annotated.parquet"))

# Limit rows for demo (e.g., 8). Use 0 or unset to disable limiting.
ROW_LIMIT = int(os.environ.get("LANCE_ROW_LIMIT", "8"))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
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


def limit_table(table: pa.Table, limit: int | None) -> pa.Table:
    """Return a slice of the table limited to `limit` rows (if > 0)."""
    try:
        if limit is None or int(limit) <= 0:
            return table
        limit = min(int(limit), len(table))
        return table.slice(0, limit)
    except Exception:
        return table


def decode_image_from_bytes(b: bytes) -> np.ndarray:
    """Decode image bytes (e.g., JPEG/PNG) into an RGB numpy array."""
    with Image.open(io.BytesIO(b)) as im:
        im = im.convert("RGB")
        return np.array(im)


def build_video_frame_packet(row: pa.Table, row_idx: int) -> VideoFramePacket:
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

    return VideoFramePacket(
        frame_data=frame_np,
        frame_number=frame_no,
        timestamp=ts,
        source_video_id=vid,
        additional_metadata={},
    )


def make_loader(table: pa.Table):
    """Create a BatchProcessor-compatible loader over a fixed Arrow table."""
    def load_func(indices: List[int]) -> List[VideoFramePacket]:
        # Take a subset table by row indices and convert to packets
        sub = table.take(pa.array(indices))
        frames: List[VideoFramePacket] = []
        for pos, row_idx in enumerate(indices):
            one = sub.slice(pos, 1)
            frames.append(build_video_frame_packet(one, row_idx))
        return frames

    return load_func



def main():
    print("Loading Lance table:", DATASET_PATH)
    table = load_lance_table(DATASET_PATH)
    # Limit for quicker demo runs
    table = limit_table(table, ROW_LIMIT)
    schema_cols = set(table.column_names)
    if IMAGE_COL not in schema_cols:
        raise ValueError(f"Column '{IMAGE_COL}' not found. Available: {sorted(schema_cols)}")

    n_rows = len(table)
    print(f"Rows: {n_rows}; image column: '{IMAGE_COL}'")

    # Indices are row numbers
    indices = list(range(n_rows))
    load_func = make_loader(table)

    # Features
    caption = cortexia.create_feature("caption")
    listing = cortexia.create_feature("listing")
    detection = cortexia.create_feature("detection")
    segmentation = cortexia.create_feature("segmentation")

    # Materialize frames once for chaining and attachment
    frames = load_func(indices)

    # 1) Caption (demonstrate BatchProcessor usage with save_func)
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

    # 3) Detection (BatchProcessor chain; uses listing prompts from metadata)
    print("Running Detection via BatchProcessor and attaching to frames...")
    det_results_map = {}
    bp3 = BatchProcessor(batch_size=4)
    bp3.load_indices(indices)

    def bp3_load(batch_indices: List[int]) -> List[VideoFramePacket]:
        return [frames_map[i] for i in batch_indices]

    def bp3_infer(fr_batch: List[VideoFramePacket], batch_indices: List[int]):
        return detection.process_batch(fr_batch)

    def bp3_save(idx: int, result):
        frames_map[idx].add_annotation_result(result)
        det_results_map[idx] = result

    _ = bp3.process_batch(load_func=bp3_load, inference_func=bp3_infer, save_func=bp3_save, filter_func=None)
    det_results = [det_results_map[i] for i in indices]

    # 4) Segmentation (BatchProcessor chain; consumes detection attached above)
    print("Running Segmentation via BatchProcessor and attaching to frames...")
    seg_results_map = {}
    bp4 = BatchProcessor(batch_size=4)
    bp4.load_indices(indices)

    def bp4_load(batch_indices: List[int]) -> List[VideoFramePacket]:
        return [frames_map[i] for i in batch_indices]

    def bp4_infer(fr_batch: List[VideoFramePacket], batch_indices: List[int]):
        return segmentation.process_batch(fr_batch)

    def bp4_save(idx: int, result):
        frames_map[idx].add_annotation_result(result)
        seg_results_map[idx] = result

    _ = bp4.process_batch(load_func=bp4_load, inference_func=bp4_infer, save_func=bp4_save, filter_func=None)
    seg_results = [seg_results_map[i] for i in indices]

    # Show a couple of examples
    for i in range(min(2, n_rows)):
        print(f"Row {i} -> caption: {cap_results[i].caption!r}")
        print(f"Row {i} -> tags: {list_results[i].tags}")
        print(
            f"Row {i} -> det: label={det_results[i].label!r}, score={det_results[i].score:.3f}, box={det_results[i].box.xyxy}"
        )
        # Segmentation may be empty depending on detection/mask models
        area = getattr(seg_results[i], "area", 0)
        print(f"Row {i} -> seg area: {area}")

    # ----------------------------------------------------------------------------
    # Write annotated results to a new table using Struct columns
    # ----------------------------------------------------------------------------
    print("Building annotated Arrow table with Struct columns...")

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

    col_caption_struct = results_to_struct_array(cap_results)
    col_tags_struct = results_to_struct_array(list_results)
    col_det_struct = results_to_struct_array(det_results)
    col_seg_struct = results_to_struct_array(seg_results)

    annotated = table
    annotated = annotated.append_column("cortexia_caption", col_caption_struct)
    annotated = annotated.append_column("cortexia_tags", col_tags_struct)
    annotated = annotated.append_column("cortexia_detection", col_det_struct)
    annotated = annotated.append_column("cortexia_segmentation", col_seg_struct)

    # Quick preview: show several rows of the new table (selected columns)
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

    # Try writing to Lance; fall back to Parquet
    wrote = False
    try:
        import lance
        out = Path(OUTPUT_LANCE)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Overwrite destination if exists by writing a fresh dataset
        if out.exists():
            # Best-effort cleanup; Lance manages versions but this is a demo
            import shutil
            shutil.rmtree(out, ignore_errors=True)
        lance.write_table(annotated, str(out))
        print(f"Annotated Lance dataset written to: {out}")
        wrote = True
    except Exception as e:
        print(f"Lance write failed or unavailable: {e}")

    if not wrote:
        outp = Path(OUTPUT_PARQUET)
        outp.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pyarrow.parquet as pq
            pq.write_table(annotated, outp)
            print(f"Annotated Parquet written to: {outp}")
        except Exception as e:
            # Final fallback: Arrow IPC
            outf = outp.with_suffix(".arrow")
            with pa.OSFile(str(outf), "wb") as sink:
                with pa.ipc.new_file(sink, annotated.schema) as writer:
                    writer.write_table(annotated)
            print(f"Annotated Arrow file written to: {outf}")


if __name__ == "__main__":
    main()
