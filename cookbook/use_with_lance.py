# %% [markdown]
# # Use Cortexia with Lance Dataset
#
# Streaming example that reads Lance batches, runs Cortexia features with
# PyTorch inference optimizations, and writes results directly to disk without
# ever storing all predictions in memory.
#
# 方案三：批内写片 + 合并更新 Lance。

from __future__ import annotations

import datetime
import io
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image

# Make local package importable when running from cookbook/
parent_path = str(Path.cwd().parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)
REPO_ROOT = parent_path

import cortexia
from cortexia.data.models.video import VideoFramePacket

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
_repo_path = Path(REPO_ROOT)

DATASET_PATH = os.environ.get(
    "LANCE_DATASET",
    str(_repo_path / "dummys" / "lance_data" / "all_in_one.lance"),
)

# Column names (customize as needed)
IMAGE_COL = os.environ.get("LANCE_IMAGE_COL", "camera_left")
VIDEO_ID_COL = os.environ.get("LANCE_VIDEO_ID_COL", None)
FRAME_NUM_COL = os.environ.get("LANCE_FRAME_NUM_COL", None)
TIMESTAMP_COL = os.environ.get("LANCE_TIMESTAMP_COL", None)
SAMPLE_ID_COL = os.environ.get("LANCE_SAMPLE_ID_COL", None)

# Output dataset root. Each round writes shards as Parquet files.
OUTPUT_ROOT = Path(
    os.environ.get(
        "LANCE_STREAM_OUTPUT",
        str(_repo_path / "dummys" / "lance_data" / "cortexia_rounds"),
    )
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Streaming configuration
BATCH_SIZE = int(os.environ.get("LANCE_STREAM_BATCH", "16"))
ROW_LIMIT = int(os.environ.get("LANCE_ROW_LIMIT", "0"))

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


TimingStats = Dict[str, List[float]]
TIMING_STATS: TimingStats = defaultdict(list)


def record_timing(name: str, elapsed: float, *, batch_id: Optional[int] = None) -> None:
    """Track and print timing information for a processing stage."""
    TIMING_STATS[name].append(elapsed)
    if batch_id is not None:
        print(f"[TIMING] {name} batch {batch_id} took {elapsed:.3f} s")
    else:
        print(f"[TIMING] {name} took {elapsed:.3f} s")


def _require_lance() -> Any:
    try:
        import lance
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Lance Python package is required for this example. Install 'lance'."
        ) from exc
    return lance


def stream_lance_batches(dataset_path: str, batch_size: int) -> Iterator[tuple[int, pa.Table]]:
    """Yield batches as Arrow tables while casting binary columns to large_binary."""
    lance = _require_lance()
    dataset = lance.dataset(dataset_path)
    batch_id = 0
    row_count = 0
    for batch in dataset.to_batches(batch_size=batch_size):
        step_start = time.time()
        if ROW_LIMIT > 0 and row_count >= ROW_LIMIT:
            break
        # Respect ROW_LIMIT by slicing batch if needed
        if ROW_LIMIT > 0 and row_count + len(batch) > ROW_LIMIT:
            batch = batch.slice(0, ROW_LIMIT - row_count)
        # Cast binary columns to large_binary to avoid Arrow offset overflows.
        columns: List[pa.Array] = []
        fields: List[pa.Field] = []
        for field in batch.schema:
            col = batch.column(field.name)
            if pa.types.is_binary(col.type):
                col = col.cast(pa.large_binary())
            columns.append(col)
            fields.append(pa.field(field.name, col.type))
        cast_batch = pa.Table.from_arrays(columns, schema=pa.schema(fields))
        record_timing("stream_lance_batches", time.time() - step_start, batch_id=batch_id)
        yield batch_id, cast_batch
        batch_id += 1
        row_count += len(cast_batch)


def decode_image_from_bytes(raw: bytes) -> np.ndarray:
    """Decode image bytes to an RGB array."""
    start = time.time()
    with Image.open(io.BytesIO(raw)) as img:
        frame = np.asarray(img.convert("RGB"))
    record_timing("decode_image_from_bytes", time.time() - start)
    return frame


def build_video_frame_packet(
    row: pa.Table,
    row_idx: int,
    *,
    video_id: Optional[str],
    frame_number: Optional[int],
    timestamp: Optional[datetime.timedelta],
    trajectory_source: Optional[Sequence[Any]] = None,
) -> VideoFramePacket:
    """Create a VideoFramePacket from a single-row table."""
    start = time.time()
    image_value = row[IMAGE_COL][0]
    if hasattr(image_value, "as_py"):
        image_value = image_value.as_py()
    frame_np = decode_image_from_bytes(image_value)

    if video_id is None:
        video_id = "lance_demo"
    if frame_number is None:
        frame_number = row_idx
    if timestamp is None:
        timestamp = datetime.timedelta(seconds=frame_number / 30.0)

    trajectory_points: List[Any] = []
    if trajectory_source:
        from cortexia.data.models.video import TrajectoryPoint

        for point in trajectory_source:
            if isinstance(point, (list, tuple)) and len(point) >= 7:
                x, y, z, qx, qy, qz, qw = point[:7]
                trajectory_points.append(
                    TrajectoryPoint(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)
                )

    packet = VideoFramePacket(
        frame_data=frame_np,
        frame_number=int(frame_number),
        timestamp=timestamp,
        source_video_id=video_id,
        additional_metadata={},
        trajectory=trajectory_points,
        current_traj_index=0,
    )
    record_timing("build_video_frame_packet", time.time() - start)
    return packet


def table_row_to_packet(
    batch: pa.Table,
    row_idx: int,
    global_row_id: int,
    *,
    default_video_id: str,
) -> VideoFramePacket:
    """Convert a batch row to a VideoFramePacket."""
    start = time.time()
    row = batch.slice(row_idx, 1)

    video_id_val: Optional[str] = default_video_id
    if VIDEO_ID_COL and VIDEO_ID_COL in row.column_names:
        raw = row[VIDEO_ID_COL][0]
        video_id_val = str(raw.as_py() if hasattr(raw, "as_py") else raw)

    frame_number_val: Optional[int] = None
    if FRAME_NUM_COL and FRAME_NUM_COL in row.column_names:
        frame_raw = row[FRAME_NUM_COL][0]
        frame_number_val = int(frame_raw.as_py() if hasattr(frame_raw, "as_py") else frame_raw)

    timestamp_val: Optional[datetime.timedelta] = None
    if TIMESTAMP_COL and TIMESTAMP_COL in row.column_names:
        ts_raw = row[TIMESTAMP_COL][0]
        ts_raw = ts_raw.as_py() if hasattr(ts_raw, "as_py") else ts_raw
        if isinstance(ts_raw, float):
            timestamp_val = datetime.timedelta(seconds=float(ts_raw))
        elif isinstance(ts_raw, int):
            timestamp_val = datetime.timedelta(milliseconds=int(ts_raw))

    odo_values: Optional[Sequence[Any]] = None
    if "odo" in row.column_names:
        odo_raw = row["odo"][0]
        odo_values = odo_raw.as_py() if hasattr(odo_raw, "as_py") else odo_raw

    packet = build_video_frame_packet(
        row,
        global_row_id,
        video_id=video_id_val,
        frame_number=frame_number_val,
        timestamp=timestamp_val,
        trajectory_source=odo_values,
    )
    record_timing("table_row_to_packet", time.time() - start)
    return packet


def run_features_on_frames(
    frames: Sequence[VideoFramePacket],
    features: MutableMapping[str, Any],
) -> Dict[str, Sequence[Any]]:
    """Run features in inference mode without storing computation graphs."""
    start = time.time()
    outputs: Dict[str, Sequence[Any]] = {}
    # Ensure Torch models stay in eval() but do not rely on eval for memory savings.
    for feature in features.values():
        model = getattr(feature, "model", None)
        if hasattr(model, "eval"):
            model.eval()

    with torch.inference_mode():
        for name, feature in features.items():
            outputs[name] = feature.process_batch(list(frames))
    record_timing("run_features_on_frames", time.time() - start)
    return outputs


def extract_sample_ids(batch: pa.Table, row_offset: int) -> List[int]:
    if SAMPLE_ID_COL and SAMPLE_ID_COL in batch.column_names:
        column = batch[SAMPLE_ID_COL]
        return [int(val.as_py() if hasattr(val, "as_py") else val) for val in column]
    return [row_offset + i for i in range(len(batch))]


def _json_default(value: Any) -> Any:
    """Helper to make complex feature results JSON serializable."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def convert_results_to_table(
    sample_ids: Sequence[int],
    round_name: str,
    batch_outputs: Dict[str, Sequence[Any]],
) -> pa.Table:
    """Convert model outputs to a narrow Arrow table suitable for Parquet writing."""
    start = time.time()
    any_failure = [False] * len(sample_ids)
    round_col = pa.array([round_name] * len(sample_ids), type=pa.string())

    caption_texts: List[str] = [""] * len(sample_ids)
    tags_json: List[str] = ["[]"] * len(sample_ids)
    detection_json: List[str] = ["[]"] * len(sample_ids)
    segmentation_json: List[str] = ["[]"] * len(sample_ids)

    if "caption" in batch_outputs:
        for idx, result in enumerate(batch_outputs["caption"]):
            text = getattr(result, "caption", "")
            caption_texts[idx] = str(text)
    if "listing" in batch_outputs:
        for idx, result in enumerate(batch_outputs["listing"]):
            tags = list(getattr(result, "tags", []) or [])
            tags_json[idx] = json.dumps(tags)
    if "detection" in batch_outputs:
        for idx, result in enumerate(batch_outputs["detection"]):
            detections = getattr(result, "detections", [])
            try:
                detection_json[idx] = json.dumps(detections, default=_json_default)
            except TypeError:
                detection_json[idx] = json.dumps(str(detections))
    if "segmentation" in batch_outputs:
        for idx, result in enumerate(batch_outputs["segmentation"]):
            masks = getattr(result, "segments", [])
            try:
                segmentation_json[idx] = json.dumps(masks, default=_json_default)
            except TypeError:
                segmentation_json[idx] = json.dumps(str(masks))

    table = pa.table(
        {
            "sample_id": pa.array(sample_ids, type=pa.int64()),
            "round": round_col,
            "any_failure": pa.array(any_failure, type=pa.bool_()),
            "cortexia_caption": pa.array(caption_texts, type=pa.string()),
            "cortexia_tags": pa.array(tags_json, type=pa.string()),
            "cortexia_detection": pa.array(detection_json, type=pa.string()),
            "cortexia_segmentation": pa.array(segmentation_json, type=pa.string()),
        }
    )
    record_timing("convert_results_to_table", time.time() - start)
    return table


def write_batch_to_parquet(round_name: str, shard_id: int, table: pa.Table) -> Path:
    """Write a single batch to a Parquet shard."""
    start = time.time()
    round_dir = OUTPUT_ROOT / f"round={round_name}"
    round_dir.mkdir(parents=True, exist_ok=True)
    file_path = round_dir / f"shard_{shard_id:06d}.parquet"
    pq.write_table(table, file_path, coerce_timestamps="ms", compression="zstd")
    record_timing("write_batch_to_parquet", time.time() - start, batch_id=shard_id)
    return file_path


def release_batch_memory(*objects: Any) -> None:
    """Explicitly drop references so Python frees memory promptly."""
    start = time.time()
    for obj in objects:
        del obj
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime
        torch.cuda.empty_cache()
    record_timing("release_batch_memory", time.time() - start)


@dataclass
class RoundConfig:
    name: str
    feature_keys: Sequence[str]


def process_round(dataset_path: str, round_cfg: RoundConfig, features: Dict[str, Any]) -> None:
    """Stream the dataset, run selected features, and write shards per batch."""
    round_start = time.time()
    active_features = {key: features[key] for key in round_cfg.feature_keys}
    default_video_id = "lance_demo"

    for shard_id, batch in stream_lance_batches(dataset_path, BATCH_SIZE):
        batch_start = time.time()
        sample_ids = extract_sample_ids(batch, shard_id * BATCH_SIZE)
        prep_start = time.time()
        frames = [
            table_row_to_packet(batch, i, sample_ids[i], default_video_id=default_video_id)
            for i in range(len(batch))
        ]
        record_timing("process_round.build_frames", time.time() - prep_start, batch_id=shard_id)
        infer_start = time.time()
        outputs = run_features_on_frames(frames, active_features)
        record_timing("process_round.run_features", time.time() - infer_start, batch_id=shard_id)
        convert_start = time.time()
        batch_table = convert_results_to_table(sample_ids, round_cfg.name, outputs)
        record_timing("process_round.convert", time.time() - convert_start, batch_id=shard_id)
        write_start = time.time()
        write_batch_to_parquet(round_cfg.name, shard_id, batch_table)
        record_timing("process_round.write", time.time() - write_start, batch_id=shard_id)

        # Drop intermediate objects to keep peak memory flat.
        release_batch_memory(batch, frames, outputs, batch_table)
        record_timing("process_round.batch", time.time() - batch_start, batch_id=shard_id)

    record_timing("process_round.total", time.time() - round_start)


def load_successful_sample_ids(round_name: str) -> Iterator[int]:
    """Yield sample_ids from prior round that completed successfully."""
    round_dir = OUTPUT_ROOT / f"round={round_name}"
    if not round_dir.exists():
        return

    dataset = ds.dataset(round_dir, format="parquet")
    scanner = dataset.scanner(filter=ds.field("any_failure") == False)  # noqa: E712
    for record_batch in scanner.to_batches():
        sample_col = record_batch["sample_id"]
        for value in sample_col:
            yield int(value.as_py())


def batched(iterable: Iterator[int], batch_size: int) -> Iterator[List[int]]:
    """Group sample_ids into bounded lists so we never hold all ids in memory."""
    buffer: List[int] = []
    for item in iterable:
        buffer.append(item)
        if len(buffer) == batch_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def process_follow_up_round(
    dataset_path: str,
    previous_round: str,
    round_cfg: RoundConfig,
    features: Dict[str, Any],
) -> None:
    """Run a follow-up round only for successful sample_ids from a prior round."""
    round_start = time.time()
    lance = _require_lance()
    dataset = lance.dataset(dataset_path)
    chunk_size = BATCH_SIZE
    shard = 0
    found_any = False
    for chunk_ids in batched(load_successful_sample_ids(previous_round), chunk_size):
        batch_start = time.time()
        found_any = True
        batch = dataset.take(chunk_ids)
        # Cast binary columns as earlier to avoid offset issues.
        cast_start = time.time()
        columns = []
        fields = []
        for field in batch.schema:
            col = batch.column(field.name)
            if pa.types.is_binary(col.type):
                col = col.cast(pa.large_binary())
            columns.append(col)
            fields.append(pa.field(field.name, col.type))
        batch = pa.Table.from_arrays(columns, schema=pa.schema(fields))
        record_timing("process_follow_up_round.cast", time.time() - cast_start, batch_id=shard)

        prep_start = time.time()
        frames = [
            table_row_to_packet(batch, i, chunk_ids[i], default_video_id="lance_demo")
            for i in range(len(batch))
        ]
        record_timing("process_follow_up_round.build_frames", time.time() - prep_start, batch_id=shard)
        infer_start = time.time()
        outputs = run_features_on_frames(frames, {key: features[key] for key in round_cfg.feature_keys})
        record_timing("process_follow_up_round.run_features", time.time() - infer_start, batch_id=shard)
        convert_start = time.time()
        result_table = convert_results_to_table(chunk_ids, round_cfg.name, outputs)
        record_timing("process_follow_up_round.convert", time.time() - convert_start, batch_id=shard)
        write_start = time.time()
        write_batch_to_parquet(round_cfg.name, shard, result_table)
        record_timing("process_follow_up_round.write", time.time() - write_start, batch_id=shard)
        shard += 1
        release_batch_memory(batch, frames, outputs, result_table)
        record_timing("process_follow_up_round.batch", time.time() - batch_start, batch_id=shard - 1)

    if not found_any:
        print(f"No samples pending for round '{round_cfg.name}'.")
    record_timing("process_follow_up_round.total", time.time() - round_start)


def _non_empty_string_mask(table: pa.Table, column: str, empty_tokens: Optional[Sequence[str]] = None) -> Optional[pa.Array]:
    if column not in table.column_names:
        return None
    values = table[column]
    mask = pc.invert(pc.is_null(values))
    if empty_tokens:
        empty_mask = None
        for token in empty_tokens:
            token_mask = pc.equal(values, pa.scalar(token, type=values.type))
            empty_mask = token_mask if empty_mask is None else pc.or_(empty_mask, token_mask)
        if empty_mask is not None:
            mask = pc.and_(mask, pc.invert(empty_mask))
    return mask


def collect_updates_for_round(round_name: str) -> Optional[pa.Table]:
    """Scan Parquet shards for a round and keep rows with non-empty predictions."""
    round_dir = OUTPUT_ROOT / f"round={round_name}"
    if not round_dir.exists():
        return None

    dataset = ds.dataset(round_dir, format="parquet")
    scanner = dataset.scanner(filter=ds.field("any_failure") == False)  # noqa: E712
    batches: List[pa.RecordBatch] = []

    for record_batch in scanner.to_batches():
        table = pa.Table.from_batches([record_batch])
        caption_mask = _non_empty_string_mask(table, "cortexia_caption", [""])
        tags_mask = _non_empty_string_mask(table, "cortexia_tags", ["[]", ""])
        detection_mask = _non_empty_string_mask(table, "cortexia_detection", ["[]", ""])
        segmentation_mask = _non_empty_string_mask(table, "cortexia_segmentation", ["[]", ""])

        combined_mask: Optional[pa.Array] = None
        for mask in (caption_mask, tags_mask, detection_mask, segmentation_mask):
            if mask is None:
                continue
            combined_mask = mask if combined_mask is None else pc.or_(combined_mask, mask)

        if combined_mask is None:
            continue
        if not pc.any(combined_mask).as_py():
            continue

        filtered = table.filter(combined_mask)
        batches.extend(filtered.to_batches())

    if not batches:
        return None

    return pa.Table.from_batches(batches)


def merge_predictions_into_lance(dataset_path: str, round_names: Sequence[str]) -> None:
    """Merge all updated prediction columns back into the Lance dataset."""
    start = time.time()
    update_tables: List[pa.Table] = []
    for round_name in round_names:
        round_table = collect_updates_for_round(round_name)
        if round_table is not None and round_table.num_rows > 0:
            update_tables.append(round_table)

    if not update_tables:
        print("No prediction updates detected; skipping Lance merge.")
        record_timing("merge_predictions_into_lance", time.time() - start)
        return

    combined_updates = pa.concat_tables(update_tables, promote=True)
    available_columns = [
        column
        for column in [
            "cortexia_caption",
            "cortexia_tags",
            "cortexia_detection",
            "cortexia_segmentation",
        ]
        if column in combined_updates.column_names
    ]

    if not available_columns:
        print("Combined updates missing prediction columns; skipping Lance merge.")
        record_timing("merge_predictions_into_lance", time.time() - start)
        return

    updates = combined_updates.select(["sample_id", *available_columns])

    lance = _require_lance()
    dataset = lance.dataset(dataset_path)

    set_expressions = {col: f"source.{col}" for col in available_columns}
    merge_result: Optional[Any] = None
    try:
        merge_result = dataset.merge(
            source=updates,
            on="sample_id",
            when_matched="source.any_failure = false",
            set=set_expressions,
        )
    except AttributeError:
        # Older Lance versions expose update() instead of merge().
        dataset.update(
            updates,
            on="sample_id",
            columns=available_columns,
        )
        print(
            "Lance merge() unavailable; used update() fallback for columns:",
            ", ".join(available_columns),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        print("Failed to merge predictions into Lance dataset:", exc)
    else:
        if merge_result is not None:
            print("Lance merge() completed:", getattr(merge_result, "summary", merge_result))

    record_timing("merge_predictions_into_lance", time.time() - start)


def print_timing_summary() -> None:
    if not TIMING_STATS:
        print("No timing data collected.")
        return

    print("\nTiming summary:")
    for name, values in TIMING_STATS.items():
        if not values:
            continue
        total = sum(values)
        avg = total / len(values)
        max_val = max(values)
        print(
            f"  {name}: count={len(values)} total={total:.3f}s avg={avg:.3f}s max={max_val:.3f}s"
        )


def main() -> None:
    global TIMING_STATS
    TIMING_STATS = defaultdict(list)
    print("Loading Cortexia features...")
    caption = cortexia.create_feature("caption")
    listing = cortexia.create_feature("listing")
    detection = cortexia.create_feature("detection")
    segmentation = cortexia.create_feature("segmentation")
    trajectory = cortexia.create_feature("trajectory")

    # Custom prompt example.
    listing.task_prompt = "List all objects in the image"

    features = {
        "caption": caption,
        "listing": listing,
        "detection": detection,
        "segmentation": segmentation,
        "trajectory": trajectory,
    }

    # Round 1: caption + listing
    round_one = RoundConfig(name="round_1_caption_listing", feature_keys=["caption", "listing"])
    process_round(DATASET_PATH, round_one, features)

    # Round 2: run detection only on samples that succeeded in round 1.
    round_two = RoundConfig(name="round_2_detection", feature_keys=["detection"])
    process_follow_up_round(DATASET_PATH, round_one.name, round_two, features)

    merge_predictions_into_lance(DATASET_PATH, [round_one.name, round_two.name])

    print("Streaming inference complete. Results stored in:", OUTPUT_ROOT)
    print_timing_summary()


if __name__ == "__main__":
    main()
