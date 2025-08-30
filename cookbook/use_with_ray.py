# %% [markdown]
# # Use Cortexia with Ray Data for Batch Processing
# 
# Example: Use Cortexia with Ray Data for distributed batch inference.
# 
# This example shows how to:
# - Read images from a Lance dataset using Ray Data
# - Use Ray Data's map_batches for distributed batch processing
# - Share models across Ray workers using Ray object store
# - Run multiple cortexia features in parallel
# - Save results back to Lance or other formats
# 
# Assumptions:
# - Ray is installed (`pip install ray[data,lance]`)
# - The Lance dataset at `dummys/lance_data/all_in_one.lance` exists
# - Image bytes are stored in the `camera_left` column
# - Sufficient compute resources (CPU/GPU) are available

# %% [markdown]
# ## Setup

# %%
# Import dependencies
from __future__ import annotations

import os
import sys
import io
import json
import math
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Dict, Union

import numpy as np
from PIL import Image

import pyarrow as pa
import ray
from ray.data import Dataset

# %%
# Set environment variables
os.environ["HF_HOME"] = "/vita-vepfs-data/fileset1/model/heng.li/huggingface"

# %%
# Make local package importable when running from cookbook/
parent_path = str(Path.cwd().parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)
REPO_ROOT = parent_path

# %% [markdown]
# ## Configuration

# %%
# Dataset paths
_repo_path = Path(REPO_ROOT)
DATASET_PATH = os.environ.get(
    "LANCE_DATASET",
    str(_repo_path / "dummys" / "lance_data" / "all_in_one.lance")
)

# Column names
IMAGE_COL = os.environ.get("LANCE_IMAGE_COL", "camera_left")
VIDEO_ID_COL = os.environ.get("LANCE_VIDEO_ID_COL", None)
FRAME_NUM_COL = os.environ.get("LANCE_FRAME_NUM_COL", None)
TIMESTAMP_COL = os.environ.get("LANCE_TIMESTAMP_COL", None)

# Output paths
OUTPUT_LANCE = os.environ.get(
    "LANCE_OUTPUT",
    str(_repo_path / "dummys" / "lance_data" / "all_in_one_ray_processed.lance")
)
OUTPUT_PARQUET = os.environ.get(
    "PARQUET_OUTPUT", 
    str(_repo_path / "dummys" / "lance_data" / "all_in_one_ray_processed.parquet")
)

# Ray configuration
RAY_NUM_CPUS = int(os.environ.get("RAY_NUM_CPUS", "4"))
RAY_NUM_GPUS = int(os.environ.get("RAY_NUM_GPUS", "0"))
RAY_BATCH_SIZE = int(os.environ.get("RAY_BATCH_SIZE", "8"))
RAY_CONCURRENCY = int(os.environ.get("RAY_CONCURRENCY", "2"))

# Limit rows for demo
ROW_LIMIT = int(os.environ.get("LANCE_ROW_LIMIT", "16"))

# %% [markdown]
# ## Initialize Ray

# %%
# Initialize Ray with appropriate resources
if not ray.is_initialized():
    ray.init(
        num_cpus=RAY_NUM_CPUS,
        num_gpus=RAY_NUM_GPUS,
        include_dashboard=False,
        ignore_reinit_error=True,
        runtime_env={
            "py_modules": [parent_path],  # Include the cortexia package
            "env_vars": {
                "PYTHONPATH": parent_path,
                "HF_HOME": "/vita-vepfs-data/fileset1/model/heng.li/huggingface"
            }
        }
    )
    print(f"Ray initialized with {RAY_NUM_CPUS} CPUs and {RAY_NUM_GPUS} GPUs")
else:
    print("Ray already initialized")

# %% [markdown]
# ## Helper Functions

# %%
def decode_image_from_bytes(b: bytes) -> np.ndarray:
    """Decode image bytes (e.g., JPEG/PNG) into an RGB numpy array."""
    with Image.open(io.BytesIO(b)) as im:
        im = im.convert("RGB")
        return np.array(im)

def build_video_frame_packet_from_dict(row_data: Dict[str, Any], row_idx: int) -> Any:
    """Construct a VideoFramePacket from a dictionary row."""
    # Import here to ensure it's available in Ray workers
    sys.path.insert(0, REPO_ROOT)
    from cortexia.data.models.video import VideoFramePacket
    
    # Image
    img_val = row_data[IMAGE_COL]
    frame_np = decode_image_from_bytes(img_val)
    
    # Video/frame/timestamp
    if VIDEO_ID_COL and VIDEO_ID_COL in row_data:
        vid = str(row_data[VIDEO_ID_COL])
    else:
        vid = "ray_demo"
    
    if FRAME_NUM_COL and FRAME_NUM_COL in row_data:
        frame_no = int(row_data[FRAME_NUM_COL])
    else:
        frame_no = int(row_idx)
    
    if TIMESTAMP_COL and TIMESTAMP_COL in row_data:
        ts_val = row_data[TIMESTAMP_COL]
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
        trajectory=[],  # Empty trajectory for simplicity
        current_traj_index=0
    )

def result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert a cortexia result to a dictionary for serialization."""
    if hasattr(result, 'to_pyarrow_struct'):
        struct = result.to_pyarrow_struct()
        return {field: struct[field].as_py() for field in struct.type.names}
    else:
        return {"error": f"Cannot serialize result of type {type(result)}"}

# %% [markdown]
# ## Ray Batch Processors for Cortexia Features

# %%
@ray.remote
class CortexiaFeatureProcessor:
    """Ray actor for processing cortexia features with shared model resources."""
    
    def __init__(self, feature_name: str, feature_config: Optional[Dict] = None):
        """Initialize the processor with a specific cortexia feature."""
        self.feature_name = feature_name
        self.feature_config = feature_config or {}
        self.feature = None
        self._initialize_feature()
    
    def _initialize_feature(self):
        """Initialize the cortexia feature."""
        try:
            # Ensure the path is set for Ray workers
            if REPO_ROOT not in sys.path:
                sys.path.insert(0, REPO_ROOT)
            
            # Import cortexia in the worker
            import cortexia
            from cortexia.data.models.video import VideoFramePacket
            
            self.feature = cortexia.create_feature(self.feature_name)
            print(f"Initialized feature: {self.feature_name}")
        except Exception as e:
            print(f"Failed to initialize feature {self.feature_name}: {e}")
            raise
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data using the initialized feature."""
        if not self.feature:
            raise RuntimeError(f"Feature {self.feature_name} not initialized")
        
        try:
            # Convert batch data to VideoFramePackets
            frames = []
            for i, row_data in enumerate(batch_data):
                frame = build_video_frame_packet_from_dict(row_data, i)
                frames.append(frame)
            
            # Process batch with cortexia feature
            results = self.feature.process_batch(frames)
            
            # Convert results to serializable format
            serialized_results = []
            for result in results:
                serialized_results.append(result_to_dict(result))
            
            return serialized_results
            
        except Exception as e:
            print(f"Error processing batch with {self.feature_name}: {e}")
            # Return error results for each item in batch
            return [{"error": str(e)} for _ in batch_data]
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the initialized feature."""
        if self.feature:
            return {
                "name": self.feature.name,
                "description": self.feature.description,
                "initialized": self.feature.is_ready()
            }
        return {"name": self.feature_name, "initialized": False}

# %% [markdown]
# ## Multi-Feature Ray Processor

# %%
class MultiFeatureRayProcessor:
    """Processor for running multiple cortexia features with Ray Data."""
    
    def __init__(self, feature_names: List[str], num_actors: int = RAY_CONCURRENCY):
        """Initialize with multiple features and Ray actors."""
        self.feature_names = feature_names
        self.num_actors = num_actors
        self.actors = {}
        self._create_actors()
    
    def _create_actors(self):
        """Create Ray actors for each feature."""
        for feature_name in self.feature_names:
            # Create multiple actors per feature for parallelism
            self.actors[feature_name] = [
                CortexiaFeatureProcessor.remote(feature_name)
                for _ in range(self.num_actors)
            ]
            print(f"Created {self.num_actors} actors for feature: {feature_name}")
    
    def process_dataset(self, dataset: Dataset, batch_size: int = RAY_BATCH_SIZE) -> Dataset:
        """Process the dataset with all features."""
        processed_dataset = dataset
        
        for feature_name in self.feature_names:
            print(f"Processing feature: {feature_name}")
            
            # Get actors for this feature
            feature_actors = self.actors[feature_name]
            
            # Define the batch processing function
            def process_feature_batch(batch: pa.Table) -> pa.Table:
                # Convert PyArrow Table to list of dicts
                batch_data = []
                for i in range(len(batch)):
                    row_data = {}
                    for col_idx, col_name in enumerate(batch.schema.names):
                        row_data[col_name] = batch[col_name][i].as_py()
                    batch_data.append(row_data)
                
                # Process with Ray actors (round-robin for simplicity)
                actor_idx = hash(str(batch_data[0].get(VIDEO_ID_COL, 0))) % len(feature_actors)
                actor = feature_actors[actor_idx]
                
                # Process batch
                results = ray.get(actor.process_batch.remote(batch_data))
                
                # Convert results back to PyArrow arrays
                result_arrays = {}
                for key in batch_data[0].keys():
                    result_arrays[key] = batch[key]
                
                # Add feature results
                result_arrays[f"cortexia_{feature_name}"] = pa.array(results)
                
                return pa.Table.from_arrays(
                    [result_arrays[col] for col in result_arrays.keys()],
                    names=list(result_arrays.keys())
                )
            
            # Apply batch processing
            processed_dataset = processed_dataset.map_batches(
                process_feature_batch,
                batch_size=batch_size,
                batch_format="pyarrow",  # Use valid batch format
                concurrency=self.num_actors,
                num_cpus=1,
                num_gpus=RAY_NUM_GPUS // len(self.feature_names) if RAY_NUM_GPUS > 0 else 0
            )
        
        return processed_dataset
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about all features."""
        info = {}
        for feature_name, actors in self.actors.items():
            info[feature_name] = ray.get(actors[0].get_feature_info.remote())
        return info

# %% [markdown]
# ## Load Data with Ray Data

# %%
print("Loading Lance dataset with Ray Data...")
try:
    # Read Lance dataset with Ray Data
    columns = [IMAGE_COL]
    if VIDEO_ID_COL:
        columns.append(VIDEO_ID_COL)
    if FRAME_NUM_COL:
        columns.append(FRAME_NUM_COL)
    if TIMESTAMP_COL:
        columns.append(TIMESTAMP_COL)
    
    dataset = ray.data.read_lance(
        DATASET_PATH,
        columns=columns
    )
    
    # Apply row limit if specified
    if ROW_LIMIT > 0:
        dataset = dataset.limit(ROW_LIMIT)
    
    print(f"Loaded dataset with {dataset.count()} rows")
    print(f"Schema: {dataset.schema()}")
    
except Exception as e:
    print(f"Failed to load Lance dataset: {e}")
    # Fallback: create a simple synthetic dataset for demonstration
    print("Creating synthetic dataset for demonstration...")
    
    # Create synthetic data
    synthetic_data = []
    for i in range(min(ROW_LIMIT, 8)):
        # Create a simple synthetic image (gradient)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        synthetic_data.append({
            IMAGE_COL: img_bytes,
            VIDEO_ID_COL: f"synthetic_video_{i % 2}" if VIDEO_ID_COL else None,
            FRAME_NUM_COL: i if FRAME_NUM_COL else None,
            TIMESTAMP_COL: float(i) if TIMESTAMP_COL else None
        })
    
    dataset = ray.data.from_items(synthetic_data)
    print(f"Created synthetic dataset with {dataset.count()} rows")

# %% [markdown]
# ## Initialize Multi-Feature Processor

# %%
# Features to process
features_to_process = ["caption", "listing", "detection", "segmentation"]

# Create multi-feature processor
processor = MultiFeatureRayProcessor(features_to_process, num_actors=RAY_CONCURRENCY)

# Get feature information
feature_info = processor.get_feature_info()
print("Feature information:")
for feature_name, info in feature_info.items():
    print(f"  {feature_name}: {info}")

# %% [markdown]
# ## Process Dataset with Ray Data

# %%
print("Starting batch processing with Ray Data...")
processed_dataset = processor.process_dataset(dataset, batch_size=RAY_BATCH_SIZE)

print("Processing completed!")
print(f"Processed dataset schema: {processed_dataset.schema()}")

# %% [markdown]
# ## Examine Results

# %%
# Show some sample results
print("Sample results:")
# Use take instead of take_batch to get individual rows
sample_rows = processed_dataset.take(3)
for i, row in enumerate(sample_rows):
    print(f"\nRow {i}:")
    # row is a dict when using take
    for feature_name in features_to_process:
        result_key = f"cortexia_{feature_name}"
        if result_key in row:
            result = row[result_key]
            print(f"  {feature_name}: {result}")

# %% [markdown]
# ## Save Results

# %%
print("Saving processed results...")

# Try to save as Lance dataset
try:
    import lance
    
    # Convert Ray Dataset to Arrow Table
    batches = list(processed_dataset.iter_batches(batch_size=1000))
    if batches:
        arrow_table = pa.concat_tables(batches)
    else:
        # Create empty table with the correct schema
        arrow_table = pa.Table.from_arrays([], schema=processed_dataset.schema())
    
    # Write to Lance
    output_path = Path(OUTPUT_LANCE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path, ignore_errors=True)
    
    lance.write_dataset(arrow_table, str(output_path))
    print(f"Results saved to Lance dataset: {output_path}")
    
except Exception as e:
    print(f"Failed to save as Lance dataset: {e}")
    
    # Fallback: save as Parquet
    try:
        output_path = Path(OUTPUT_PARQUET)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_dataset.write_parquet(str(output_path))
        print(f"Results saved to Parquet files: {output_path}")
        
    except Exception as e2:
        print(f"Failed to save as Parquet: {e2}")

# %% [markdown]
# ## Performance Summary

# %%
print("\n=== Processing Summary ===")
print(f"Total rows processed: {processed_dataset.count()}")
print(f"Features processed: {', '.join(features_to_process)}")
print(f"Ray configuration:")
print(f"  - CPUs: {RAY_NUM_CPUS}")
print(f"  - GPUs: {RAY_NUM_GPUS}")
print(f"  - Batch size: {RAY_BATCH_SIZE}")
print(f"  - Concurrency: {RAY_CONCURRENCY}")
print(f"  - Row limit: {ROW_LIMIT}")

# %% [markdown]
# ## Cleanup

# %%
# Shutdown Ray
if ray.is_initialized():
    ray.shutdown()
    print("Ray shutdown completed")

# %%
print("Ray batch processing example completed!")