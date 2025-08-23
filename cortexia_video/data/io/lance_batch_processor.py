"""Enhanced batch processor with Lance dataset integration.

This module extends the generic batch processor to support Lance datasets
with GPU-optimized operations for video frame processing.
"""

import gc
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union

try:
    import lance
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

from .lance_adapter import LanceAdapter
from .lance_mixin import LanceMixin
from ..models.video import (
    AnnotatedFramePacket,
    TaggedFramePacket,
    VideoFramePacket,
)

T = TypeVar('T')


class LanceBatchProcessor(LanceMixin):
    """Enhanced batch processor with Lance dataset integration.
    
    Provides GPU-optimized batch processing capabilities for Lance datasets,
    with memory-efficient streaming and automatic result persistence.
    """
    
    def __init__(self, batch_size: int = 4, enable_gpu_optimization: bool = True):
        """Initialize Lance-enabled batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
            enable_gpu_optimization: Whether to enable GPU memory optimizations
        """
        super().__init__()
        self.batch_size = batch_size
        self.enable_gpu_optimization = enable_gpu_optimization
        self._lance_adapter = LanceAdapter()
    
    def _ensure_lance_available(self):
        """Check if Lance is available."""
        if not LANCE_AVAILABLE:
            raise ImportError(
                "Lance is not available. Install with: pip install pylance"
            )
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory if optimization is enabled."""
        if self.enable_gpu_optimization:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        gc.collect()
    
    def process_frames_from_lance(
        self,
        input_dataset_path: Union[str, Path],
        output_dataset_path: Union[str, Path],
        inference_func: Callable[[List[VideoFramePacket]], List[AnnotatedFramePacket]],
        filter_expr: Optional[str] = None,
        frame_numbers: Optional[List[int]] = None,
        save_mode: str = "append",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Process video frames from Lance dataset with batch inference.
        
        Args:
            input_dataset_path: Path to input Lance dataset
            output_dataset_path: Path to output Lance dataset  
            inference_func: Function that takes frame packets and returns annotated packets
            filter_expr: SQL-like filter expression for input selection
            frame_numbers: Specific frame numbers to process
            save_mode: Save mode for output ("append", "overwrite", "create")
            progress_callback: Optional callback for progress tracking (current, total)
        """
        self._ensure_lance_available()
        
        # Count total frames for progress tracking
        input_dataset = lance.dataset(str(input_dataset_path))
        
        # Apply filters to get total count
        query = input_dataset.to_table()
        if filter_expr:
            query = query.filter(pa.compute.expression(filter_expr))
        if frame_numbers is not None:
            frame_filter = pa.compute.is_in(
                pa.compute.field("frame_number"),
                pa.array(frame_numbers)
            )
            query = query.filter(frame_filter)
        
        total_frames = len(query)
        processed_frames = 0
        
        print(f"Processing {total_frames} frames in batches of {self.batch_size}")
        
        # Process in batches
        batch = []
        for frame_packet in self.load_frames_from_lance(
            input_dataset_path,
            frame_numbers=frame_numbers,
            filter_expr=filter_expr
        ):
            batch.append(frame_packet)
            
            if len(batch) >= self.batch_size:
                # Process batch
                try:
                    annotated_batch = inference_func(batch)
                    
                    # Save results
                    self.save_annotated_frames_to_lance(
                        annotated_batch,
                        output_dataset_path,
                        mode=save_mode if processed_frames == 0 else "append"
                    )
                    
                    processed_frames += len(batch)
                    
                    if progress_callback:
                        progress_callback(processed_frames, total_frames)
                    
                    print(f"Processed batch: {processed_frames}/{total_frames} frames")
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
                finally:
                    # Clean up memory
                    batch.clear()
                    self._cleanup_gpu_memory()
        
        # Process remaining frames
        if batch:
            try:
                annotated_batch = inference_func(batch)
                self.save_annotated_frames_to_lance(
                    annotated_batch,
                    output_dataset_path, 
                    mode="append"
                )
                processed_frames += len(batch)
                
                if progress_callback:
                    progress_callback(processed_frames, total_frames)
                
                print(f"Processed final batch: {processed_frames}/{total_frames} frames")
                
            except Exception as e:
                print(f"Error processing final batch: {e}")
            finally:
                batch.clear()
                self._cleanup_gpu_memory()
    
    def process_annotated_frames_from_lance(
        self,
        input_dataset_path: Union[str, Path],
        output_dataset_path: Union[str, Path],
        gate_func: Callable[[List[AnnotatedFramePacket]], List[TaggedFramePacket]],
        filter_expr: Optional[str] = None,
        frame_numbers: Optional[List[int]] = None,
        save_mode: str = "append",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Process annotated frames from Lance dataset with gate evaluation.
        
        Args:
            input_dataset_path: Path to input Lance dataset
            output_dataset_path: Path to output Lance dataset
            gate_func: Function that takes annotated packets and returns tagged packets
            filter_expr: SQL-like filter expression
            frame_numbers: Specific frame numbers to process
            save_mode: Save mode for output
            progress_callback: Optional progress callback
        """
        self._ensure_lance_available()
        
        # Count total frames
        input_dataset = lance.dataset(str(input_dataset_path))
        query = input_dataset.to_table()
        if filter_expr:
            query = query.filter(pa.compute.expression(filter_expr))
        if frame_numbers is not None:
            frame_filter = pa.compute.is_in(
                pa.compute.field("frame_number"),
                pa.array(frame_numbers)
            )
            query = query.filter(frame_filter)
        
        total_frames = len(query)
        processed_frames = 0
        
        print(f"Processing {total_frames} annotated frames through gates in batches of {self.batch_size}")
        
        # Process in batches
        batch = []
        for annotated_packet in self.load_annotated_frames_from_lance(
            input_dataset_path,
            frame_numbers=frame_numbers,
            filter_expr=filter_expr
        ):
            batch.append(annotated_packet)
            
            if len(batch) >= self.batch_size:
                try:
                    tagged_batch = gate_func(batch)
                    
                    # Save results
                    self.save_tagged_frames_to_lance(
                        tagged_batch,
                        output_dataset_path,
                        mode=save_mode if processed_frames == 0 else "append"
                    )
                    
                    processed_frames += len(batch)
                    
                    if progress_callback:
                        progress_callback(processed_frames, total_frames)
                    
                    print(f"Processed gate batch: {processed_frames}/{total_frames} frames")
                    
                except Exception as e:
                    print(f"Error processing gate batch: {e}")
                    continue
                finally:
                    batch.clear()
                    self._cleanup_gpu_memory()
        
        # Process remaining frames
        if batch:
            try:
                tagged_batch = gate_func(batch)
                self.save_tagged_frames_to_lance(
                    tagged_batch,
                    output_dataset_path,
                    mode="append"
                )
                processed_frames += len(batch)
                
                if progress_callback:
                    progress_callback(processed_frames, total_frames)
                
                print(f"Processed final gate batch: {processed_frames}/{total_frames} frames")
                
            except Exception as e:
                print(f"Error processing final gate batch: {e}")
            finally:
                batch.clear()
                self._cleanup_gpu_memory()
    
    def process_streaming_pipeline(
        self,
        input_dataset_path: Union[str, Path],
        output_dataset_path: Union[str, Path],
        pipeline_funcs: List[Callable],
        filter_expr: Optional[str] = None,
        frame_numbers: Optional[List[int]] = None,
        save_mode: str = "append",
        intermediate_save: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Process frames through a streaming pipeline of functions.
        
        Args:
            input_dataset_path: Path to input Lance dataset
            output_dataset_path: Path to final output Lance dataset
            pipeline_funcs: List of processing functions to apply in sequence
            filter_expr: SQL-like filter expression
            frame_numbers: Specific frame numbers to process
            save_mode: Save mode for output
            intermediate_save: Whether to save intermediate results
            progress_callback: Optional progress callback
        """
        self._ensure_lance_available()
        
        if not pipeline_funcs:
            raise ValueError("At least one pipeline function is required")
        
        # Count total frames
        input_dataset = lance.dataset(str(input_dataset_path))
        query = input_dataset.to_table()
        if filter_expr:
            query = query.filter(pa.compute.expression(filter_expr))
        if frame_numbers is not None:
            frame_filter = pa.compute.is_in(
                pa.compute.field("frame_number"),
                pa.array(frame_numbers)
            )
            query = query.filter(frame_filter)
        
        total_frames = len(query)
        processed_frames = 0
        
        print(f"Processing {total_frames} frames through {len(pipeline_funcs)}-stage pipeline")
        
        # Start with video frames
        batch = []
        for frame_packet in self.load_frames_from_lance(
            input_dataset_path,
            frame_numbers=frame_numbers,
            filter_expr=filter_expr
        ):
            batch.append(frame_packet)
            
            if len(batch) >= self.batch_size:
                try:
                    # Apply pipeline functions in sequence
                    current_batch = batch
                    for i, func in enumerate(pipeline_funcs):
                        current_batch = func(current_batch)
                        
                        # Optional intermediate save
                        if intermediate_save and i < len(pipeline_funcs) - 1:
                            intermediate_path = f"{output_dataset_path}_stage_{i+1}"
                            if isinstance(current_batch[0], AnnotatedFramePacket):
                                self.save_annotated_frames_to_lance(
                                    current_batch, intermediate_path, mode="append"
                                )
                            elif isinstance(current_batch[0], TaggedFramePacket):
                                self.save_tagged_frames_to_lance(
                                    current_batch, intermediate_path, mode="append"
                                )
                    
                    # Save final results
                    if isinstance(current_batch[0], AnnotatedFramePacket):
                        self.save_annotated_frames_to_lance(
                            current_batch,
                            output_dataset_path,
                            mode=save_mode if processed_frames == 0 else "append"
                        )
                    elif isinstance(current_batch[0], TaggedFramePacket):
                        self.save_tagged_frames_to_lance(
                            current_batch,
                            output_dataset_path,
                            mode=save_mode if processed_frames == 0 else "append"
                        )
                    
                    processed_frames += len(batch)
                    
                    if progress_callback:
                        progress_callback(processed_frames, total_frames)
                    
                    print(f"Processed pipeline batch: {processed_frames}/{total_frames} frames")
                    
                except Exception as e:
                    print(f"Error processing pipeline batch: {e}")
                    continue
                finally:
                    batch.clear()
                    self._cleanup_gpu_memory()
        
        # Process remaining frames
        if batch:
            try:
                current_batch = batch
                for i, func in enumerate(pipeline_funcs):
                    current_batch = func(current_batch)
                    
                    if intermediate_save and i < len(pipeline_funcs) - 1:
                        intermediate_path = f"{output_dataset_path}_stage_{i+1}"
                        if isinstance(current_batch[0], AnnotatedFramePacket):
                            self.save_annotated_frames_to_lance(
                                current_batch, intermediate_path, mode="append"
                            )
                        elif isinstance(current_batch[0], TaggedFramePacket):
                            self.save_tagged_frames_to_lance(
                                current_batch, intermediate_path, mode="append"
                            )
                
                # Save final results
                if isinstance(current_batch[0], AnnotatedFramePacket):
                    self.save_annotated_frames_to_lance(
                        current_batch, output_dataset_path, mode="append"
                    )
                elif isinstance(current_batch[0], TaggedFramePacket):
                    self.save_tagged_frames_to_lance(
                        current_batch, output_dataset_path, mode="append"
                    )
                
                processed_frames += len(batch)
                
                if progress_callback:
                    progress_callback(processed_frames, total_frames)
                
                print(f"Processed final pipeline batch: {processed_frames}/{total_frames} frames")
                
            except Exception as e:
                print(f"Error processing final pipeline batch: {e}")
            finally:
                batch.clear()
                self._cleanup_gpu_memory()
    
    def get_processing_stats(
        self, 
        dataset_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Get processing statistics for a Lance dataset.
        
        Args:
            dataset_path: Path to Lance dataset
            
        Returns:
            Dictionary containing processing statistics
        """
        stats = self.get_lance_dataset_info(dataset_path)
        
        if "error" not in stats:
            try:
                dataset = lance.dataset(str(dataset_path))
                table = dataset.to_table()
                
                # Add processing-specific stats
                stats.update({
                    "total_frames": len(table),
                    "unique_videos": len(set(table["source_video_id"].to_pylist())),
                    "frame_number_range": {
                        "min": int(pa.compute.min(table["frame_number"]).as_py()),
                        "max": int(pa.compute.max(table["frame_number"]).as_py())
                    },
                    "has_annotations": "caption" in [field.name for field in table.schema],
                    "has_gate_results": "gate_passes" in [field.name for field in table.schema],
                })
                
                # Check for processing completeness
                if "caption" in [field.name for field in table.schema]:
                    non_null_captions = pa.compute.count(
                        table["caption"], mode="only_valid"
                    ).as_py()
                    stats["annotation_completeness"] = non_null_captions / len(table)
                
                if "gate_passes" in [field.name for field in table.schema]:
                    passing_frames = pa.compute.sum(table["gate_passes"]).as_py()
                    stats["gate_pass_rate"] = passing_frames / len(table)
                
            except Exception as e:
                stats["processing_stats_error"] = str(e)
        
        return stats


def create_lance_batch_processor(
    batch_size: int = 4,
    enable_gpu_optimization: bool = True
) -> LanceBatchProcessor:
    """Factory function to create a LanceBatchProcessor instance.
    
    Args:
        batch_size: Number of items to process in each batch
        enable_gpu_optimization: Whether to enable GPU memory optimizations
        
    Returns:
        LanceBatchProcessor instance
    """
    return LanceBatchProcessor(
        batch_size=batch_size,
        enable_gpu_optimization=enable_gpu_optimization
    )