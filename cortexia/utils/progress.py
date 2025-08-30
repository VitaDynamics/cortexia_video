"""Progress tracking utilities for Cortexia Video SDK"""

import time
from typing import Any, Callable, Dict, Optional, Union

from ..api.exceptions import ProcessingError


class ProgressTracker:
    """Utility class for tracking processing progress"""
    
    def __init__(
        self,
        total: int,
        description: str = "Processing",
        unit: str = "items",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the processing task
            unit: Unit of measurement (e.g., "frames", "files")
            callback: Optional callback function for progress updates
        """
        self.total = total
        self.current = 0
        self.description = description
        self.unit = unit
        self.callback = callback
        self.start_time = None
        self.end_time = None
        self.paused = False
        self.paused_time = 0.0
        self.pause_start_time = None
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish()
    
    def start(self):
        """Start progress tracking"""
        if self.start_time is None:
            self.start_time = time.time()
    
    def update(
        self,
        increment: int = 1,
        message: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        Update progress.
        
        Args:
            increment: Number of items processed
            message: Optional status message
            extra_info: Additional progress information
        """
        if self.start_time is None:
            self.start()
        
        self.current += increment
        
        # Ensure current doesn't exceed total
        if self.current > self.total:
            self.current = self.total
        
        # Trigger callback if provided
        if self.callback:
            progress_info = self.get_progress_info(message, extra_info)
            self.callback(progress_info)
    
    def set_current(self, current: int, message: Optional[str] = None):
        """
        Set current progress directly.
        
        Args:
            current: Current number of items processed
            message: Optional status message
        """
        if self.start_time is None:
            self.start()
        
        self.current = max(0, min(current, self.total))
        
        # Trigger callback if provided
        if self.callback:
            progress_info = self.get_progress_info(message)
            self.callback(progress_info)
    
    def pause(self):
        """Pause progress tracking"""
        if not self.paused and self.start_time is not None:
            self.paused = True
            self.pause_start_time = time.time()
    
    def resume(self):
        """Resume progress tracking"""
        if self.paused and self.pause_start_time is not None:
            self.paused_time += time.time() - self.pause_start_time
            self.paused = False
            self.pause_start_time = None
    
    def finish(self):
        """Finish progress tracking"""
        if self.start_time is None:
            self.start()
        
        self.end_time = time.time()
        self.current = self.total
        
        # Trigger final callback if provided
        if self.callback:
            progress_info = self.get_progress_info("Completed")
            self.callback(progress_info)
    
    def get_progress_info(
        self,
        message: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Args:
            message: Optional status message
            extra_info: Additional progress information
            
        Returns:
            Dictionary with progress information
        """
        current_time = time.time()
        
        # Calculate elapsed time
        if self.end_time is not None:
            elapsed = self.end_time - self.start_time - self.paused_time
        elif self.start_time is not None:
            if self.paused:
                elapsed = self.pause_start_time - self.start_time - self.paused_time
            else:
                elapsed = current_time - self.start_time - self.paused_time
        else:
            elapsed = 0.0
        
        # Calculate percentage
        percentage = (self.current / self.total * 100.0) if self.total > 0 else 0.0
        
        # Calculate remaining time estimate
        if self.current > 0 and elapsed > 0 and not self.finished:
            rate = self.current / elapsed
            remaining_items = self.total - self.current
            remaining_time = remaining_items / rate
        else:
            remaining_time = None
        
        # Build progress info
        progress_info = {
            "current": self.current,
            "total": self.total,
            "percentage": percentage,
            "elapsed": elapsed,
            "remaining": remaining_time,
            "rate": self.current / elapsed if elapsed > 0 else 0.0,
            "description": self.description,
            "unit": self.unit,
            "finished": self.finished,
            "paused": self.paused,
            "message": message or f"Processed {self.current} of {self.total} {self.unit}"
        }
        
        # Add extra information if provided
        if extra_info:
            progress_info.update(extra_info)
        
        return progress_info
    
    @property
    def finished(self) -> bool:
        """Check if progress is finished"""
        return self.current >= self.total and self.end_time is not None
    
    @property
    def percentage(self) -> float:
        """Get current progress percentage"""
        return (self.current / self.total * 100.0) if self.total > 0 else 0.0
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.end_time is not None:
            return self.end_time - self.start_time - self.paused_time
        elif self.start_time is not None:
            if self.paused:
                return self.pause_start_time - self.start_time - self.paused_time
            else:
                return time.time() - self.start_time - self.paused_time
        return 0.0
    
    @property
    def remaining(self) -> Optional[float]:
        """Get estimated remaining time in seconds"""
        if self.current > 0 and self.elapsed > 0 and not self.finished:
            rate = self.current / self.elapsed
            remaining_items = self.total - self.current
            return remaining_items / rate
        return None


class MultiProgressTracker:
    """Utility class for tracking multiple progress tasks"""
    
    def __init__(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize multi-progress tracker.
        
        Args:
            callback: Optional callback function for progress updates
        """
        self.trackers: Dict[str, ProgressTracker] = {}
        self.callback = callback
    
    def add_tracker(
        self,
        name: str,
        total: int,
        description: Optional[str] = None,
        unit: str = "items"
    ) -> ProgressTracker:
        """
        Add a new progress tracker.
        
        Args:
            name: Unique name for the tracker
            total: Total number of items
            description: Optional description (defaults to name)
            unit: Unit of measurement
            
        Returns:
            Created ProgressTracker
        """
        if name in self.trackers:
            raise ValueError(f"Progress tracker '{name}' already exists")
        
        tracker = ProgressTracker(
            total=total,
            description=description or name,
            unit=unit,
            callback=self._tracker_callback
        )
        
        self.trackers[name] = tracker
        return tracker
    
    def get_tracker(self, name: str) -> ProgressTracker:
        """
        Get a progress tracker by name.
        
        Args:
            name: Name of the tracker
            
        Returns:
            ProgressTracker
            
        Raises:
            KeyError: If tracker not found
        """
        if name not in self.trackers:
            raise KeyError(f"Progress tracker '{name}' not found")
        
        return self.trackers[name]
    
    def update_tracker(
        self,
        name: str,
        increment: int = 1,
        message: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        Update a specific progress tracker.
        
        Args:
            name: Name of the tracker
            increment: Number of items processed
            message: Optional status message
            extra_info: Additional progress information
        """
        if name not in self.trackers:
            raise KeyError(f"Progress tracker '{name}' not found")
        
        self.trackers[name].update(increment, message, extra_info)
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """
        Get overall progress across all trackers.
        
        Returns:
            Dictionary with overall progress information
        """
        if not self.trackers:
            return {
                "current": 0,
                "total": 0,
                "percentage": 0.0,
                "finished": True,
                "trackers": {}
            }
        
        total_current = sum(tracker.current for tracker in self.trackers.values())
        total_total = sum(tracker.total for tracker in self.trackers.values())
        all_finished = all(tracker.finished for tracker in self.trackers.values())
        
        # Calculate overall percentage
        percentage = (total_current / total_total * 100.0) if total_total > 0 else 0.0
        
        # Get individual tracker info
        trackers_info = {}
        for name, tracker in self.trackers.items():
            trackers_info[name] = tracker.get_progress_info()
        
        return {
            "current": total_current,
            "total": total_total,
            "percentage": percentage,
            "finished": all_finished,
            "trackers": trackers_info
        }
    
    def _tracker_callback(self, progress_info: Dict[str, Any]):
        """Internal callback for individual tracker updates"""
        if self.callback:
            overall_progress = self.get_overall_progress()
            overall_progress["updated_tracker"] = progress_info["description"]
            overall_progress["tracker_info"] = progress_info
            self.callback(overall_progress)
    
    def start_all(self):
        """Start all progress trackers"""
        for tracker in self.trackers.values():
            tracker.start()
    
    def finish_all(self):
        """Finish all progress trackers"""
        for tracker in self.trackers.values():
            tracker.finish()


class ProgressCallback:
    """Callback class for handling progress updates"""
    
    def __init__(
        self,
        print_progress: bool = True,
        log_file: Optional[str] = None,
        update_frequency: float = 1.0
    ):
        """
        Initialize progress callback.
        
        Args:
            print_progress: Whether to print progress to console
            log_file: Optional file path for progress logging
            update_frequency: Minimum time between updates (seconds)
        """
        self.print_progress = print_progress
        self.log_file = log_file
        self.update_frequency = update_frequency
        self.last_update_time = 0.0
    
    def __call__(self, progress_info: Dict[str, Any]):
        """
        Handle progress update.
        
        Args:
            progress_info: Progress information dictionary
        """
        current_time = time.time()
        
        # Check update frequency
        if current_time - self.last_update_time < self.update_frequency:
            return
        
        self.last_update_time = current_time
        
        # Print progress if enabled
        if self.print_progress:
            self._print_progress(progress_info)
        
        # Log progress if file specified
        if self.log_file:
            self._log_progress(progress_info)
    
    def _print_progress(self, progress_info: Dict[str, Any]):
        """Print progress to console"""
        percentage = progress_info["percentage"]
        current = progress_info["current"]
        total = progress_info["total"]
        description = progress_info["description"]
        unit = progress_info["unit"]
        message = progress_info.get("message", "")
        
        # Build progress bar
        bar_width = 50
        filled = int(bar_width * percentage / 100)
        bar = "=" * filled + "-" * (bar_width - filled)
        
        # Format time information
        elapsed = self._format_time(progress_info["elapsed"])
        remaining = progress_info.get("remaining")
        remaining_str = self._format_time(remaining) if remaining is not None else "Unknown"
        
        # Print progress line
        progress_line = (
            f"\r{description}: [{bar}] {percentage:5.1f}% "
            f"({current}/{total} {unit}) "
            f"Elapsed: {elapsed}, Remaining: {remaining_str}"
        )
        
        if message:
            progress_line += f" - {message}"
        
        print(progress_line, end="", flush=True)
        
        # Print newline if finished
        if progress_info["finished"]:
            print()
    
    def _log_progress(self, progress_info: Dict[str, Any]):
        """Log progress to file"""
        import json
        from datetime import datetime
        
        # Add timestamp
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "progress": progress_info
        }
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _format_time(self, seconds: Optional[float]) -> str:
        """Format time in seconds to human-readable string"""
        if seconds is None:
            return "Unknown"
        
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)}m {remaining_seconds:.1f}s"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(remaining_minutes)}m"


def create_progress_callback(
    print_progress: bool = True,
    log_file: Optional[str] = None,
    update_frequency: float = 1.0
) -> ProgressCallback:
    """
    Create a progress callback.
    
    Args:
        print_progress: Whether to print progress to console
        log_file: Optional file path for progress logging
        update_frequency: Minimum time between updates (seconds)
        
    Returns:
        ProgressCallback instance
    """
    return ProgressCallback(print_progress, log_file, update_frequency)