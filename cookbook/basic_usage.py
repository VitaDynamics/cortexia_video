# %% [markdown]
# # Use Cortexia SDK 

# %%
%load_ext autoreload
%autoreload 2

# append cortexia to system PATH.
import sys
from pathlib import Path

parent_path = str(Path.cwd().parent)
if parent_path not in sys.path:
    sys.path.append(parent_path)


# %%
import cortexia

# %%

# List all features for annotation
features = cortexia.list_features()
print("Available features for annotation:")
print(features)

# List all gates (for filter and gates, works as annotations)
gates = cortexia.list_gates()
print("Available gates (for filter and annotation):")
print(gates)

# %% [markdown]
# ## Create feature 
# 
# - Use feature name above to create feature for annotation usage. 

# %%
caption_feature = cortexia.create_feature("caption")

# %%
# Each feature have its own output schema for output checking.
print(caption_feature.output_schema)

# %%
# We use lazy init for each feature and module. Module only init when actually be used. 
print(caption_feature.is_ready())

# Or we can use _initialize()
caption_feature._initialize()

# Then we check again 
print(caption_feature.is_ready())

# %% [markdown]
# ## Single Feature Inference
# 
# Below we show how to use a single feature (e.g. caption) on one or multiple `VideoFramePacket` objects manually without the batch helper.
# 

# %%
import datetime, numpy as np
from cortexia.data.models.video import VideoFramePacket

# Create a dummy frame
frame = VideoFramePacket(
    frame_data=np.random.randint(0,255,(224,224,3),dtype=np.uint8),
    frame_number=0,
    timestamp=datetime.timedelta(seconds=0),
    source_video_id="demo_video"
)

# Use an already created caption_feature (from earlier cell)
if not caption_feature.is_ready():
    caption_feature._initialize()

single_result = caption_feature.process_batch([frame])[0]
print(single_result)

# Attach annotation back to frame (optional for later chaining)
frame.add_annotation_result(single_result)
print("Frame annotations keys:", list(frame.annotations.keys()))

# %% [markdown]
# ## Gate Usage
# 
# Gates compute measurements (e.g., blur, entropy) rather than semantic annotations. They also use `process_batch`.
# We'll create a gate instance and run it on the same frame.
# 

# %%
# Example: use a blur gate (replace 'blur' with an available gate name from list_gates())
blur_gate_cls = cortexia.get_gate("blur")  # get class
blur_gate = blur_gate_cls()                 # instantiate

blur_result = blur_gate.process_batch([frame])[0]
print(blur_result)
frame.add_annotation_result(blur_result)
print("Updated annotations keys:", list(frame.annotations.keys()))

# %% [markdown]
# ## Batch Processing with `process_batch_with_processor`
# 
# The helper wraps `BatchProcessor` to load frames lazily. Provide:
# - indices: list of references (here we'll simulate simple integers)
# - load_func: given a sublist of indices returns a list of `VideoFramePacket`
# - processor: a feature or gate instance (must implement `process_batch`)
# 
# We'll demo with the caption feature over 8 dummy frames in batches of 4.
# 

# %%
from cortexia.api.cortexia import process_batch_with_processor

# Simulated indices (e.g., could be frame numbers or file paths)
indices = list(range(8))

import datetime
import numpy as np

def load_func(batch_indices):
    frames = []
    for idx in batch_indices:
        frames.append(
            VideoFramePacket(
                frame_data=np.random.randint(0,255,(224,224,3),dtype=np.uint8),
                frame_number=idx,
                timestamp=datetime.timedelta(seconds=idx/30),
                source_video_id="demo_video"
            )
        )
    return frames

# Reuse caption_feature
results = process_batch_with_processor(
    indices=indices,
    load_func=load_func,
    processor=caption_feature,
    batch_size=4,
    filter_func=lambda i: i % 2 == 0  # only even indices
)

print(f"Collected {len(results)} results (expected 4 even indices)")
for r in results[:2]:
    print(r)


# %% [markdown]
# ### Using the same helper with a Gate
# 
# You can pass any gate instance to the same helper to compute measurements batch-wise.
# 

# %%
gate_batch_results = process_batch_with_processor(
    indices=indices,
    load_func=load_func,
    processor=blur_gate,
    batch_size=4,
    filter_func=None  # process all indices
)
print(f"Gate results count: {len(gate_batch_results)}")
print(gate_batch_results[0])

# %% [markdown]
# ## Summary
# 
# You learned how to:
# 1. List and instantiate features and gates.
# 2. Run a single feature and gate on a frame.
# 3. Attach results back to a `VideoFramePacket`.
# 4. Use the `process_batch_with_processor` helper for batched feature/gate processing with custom loading and filtering.
# 
# Adapt the `load_func` to read real frames from videos or datasets (e.g., using decord, OpenCV, or Arrow/Lance datasets).


