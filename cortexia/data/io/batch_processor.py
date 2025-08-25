"""Generic batch processor SDK for cortexia.

This module provides a generic batch processing interface that can be used
for any type of batch inference tasks. It separates concerns between loading,
inference, saving, and filtering operations.
"""

#TODO: This is a lazy abstraction for io related things. But currently this repo do not manupilate different io I think. We also do not need this. IO should work with user instead of being encapsulated in this repo.

import gc
from typing import Any, Callable, List, Optional, TypeVar

T = TypeVar('T')

class BatchProcessor:
    """Generic batch processor for inference tasks.
    
    This class provides a flexible interface for batch processing any type of data
    by accepting custom load, inference, save, and filter functions.
    """

    def __init__(self, batch_size: int = 4):
        """Initialize batch processor.

        Args:
            batch_size: Number of items to process in each batch
        """
        self.batch_size = batch_size
        self.indices: List[T] = []

    def load_indices(self, indices: List[T]) -> None:
        """Store indices for batch processing.

        Args:
            indices: List of indices (paths, database IDs, etc.) to store
        """
        self.indices = indices.copy()

    def process_batch(
        self,
        load_func: Callable[[List[T]], List[Any]],
        inference_func: Callable[[List[Any], List[T]], List[Any]],
        save_func: Callable[[T, Any], None],
        filter_func: Optional[Callable[[T], bool]] = None,
    ) -> None:
        """Process items in batches using provided functions.

        Args:
            load_func: Function that takes indices and returns loaded objects
            inference_func: Function that takes (objects, indices) and returns inference results
            save_func: Function that takes (index, result) and saves the result
            filter_func: Optional function to filter which indices to process
        """
        # Filter indices if filter function provided
        indices_to_process = self.indices.copy()
        if filter_func:
            indices_to_process = [idx for idx in indices_to_process if filter_func(idx)]

        if not indices_to_process:
            print("No indices to process after filtering")
            return

        # Process in batches
        total_batches = (len(indices_to_process) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(indices_to_process), self.batch_size):
            batch_indices = indices_to_process[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_indices)} items)")

            # Load objects for this batch only
            try:
                batch_objects = load_func(batch_indices)
                if not batch_objects:
                    print(f"No valid objects loaded for batch {batch_num}")
                    continue
                    
                print(f"Loaded {len(batch_objects)} objects")
                
            except Exception as e:
                print(f"Failed to load batch {batch_num}: {e}")
                continue

            try:
                # Run inference on batch
                print(f"Running inference on {len(batch_objects)} objects")
                results = inference_func(batch_objects, batch_indices)

                if len(results) != len(batch_indices):
                    print(f"Warning: Expected {len(batch_indices)} results, got {len(results)}")

                print(f"Generated {len(results)} results")

                # Save results for each object
                for idx, result in zip(batch_indices, results):
                    try:
                        save_func(idx, result)
                    except Exception as e:
                        print(f"Failed to save result for {idx}: {e}")

            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                # Continue with next batch
            finally:
                # Clean up batch objects from memory
                self._cleanup_objects(batch_objects)
                batch_objects = None
                
                # Force garbage collection after each batch
                gc.collect()

    def _cleanup_objects(self, objects: Any) -> None:
        """Clean up objects by calling close() if available."""
        if hasattr(objects, '__iter__'):
            try:
                for obj in objects:
                    if hasattr(obj, 'close'):
                        obj.close()
            except (TypeError, AttributeError):
                # objects might not be iterable or closeable
                pass

    def clear_buffer(self) -> None:
        """Clear the index buffer."""
        self.indices.clear()
        gc.collect()

    def get_indices(self) -> List[T]:
        """Get list of stored indices."""
        return self.indices.copy()

    def __len__(self) -> int:
        """Return number of stored indices."""
        return len(self.indices)

    def __repr__(self) -> str:
        """Return string representation of the processor."""
        return f"BatchProcessor(batch_size={self.batch_size}, indices_count={len(self.indices)})"


def create_batch_processor(batch_size: int = 4) -> BatchProcessor:
    """Factory function to create a BatchProcessor instance.
    
    Args:
        batch_size: Number of items to process in each batch
        
    Returns:
        BatchProcessor instance
    """
    return BatchProcessor(batch_size=batch_size)