"""
Cortexia Video Gates Module

This module provides various gate implementations for filtering video frames
based on different criteria such as blur, entropy, content similarity, etc.

Gates are filters that analyze video frames and determine whether they should
pass through (return True) or be filtered out (return False).

Example usage:

    # Using individual gate classes
    from cortexia_video.gates import BlurGate
    blur_gate = BlurGate(threshold=100.0)
    
    # Using the factory
    from cortexia_video.gates import GateFactory
    blur_gate = GateFactory.create('blur', threshold=100.0)
    
    # Using convenience functions
    from cortexia_video.gates import create_blur_gate
    blur_gate = create_blur_gate(threshold=100.0)
"""

# Import all gate classes
from .base_gate import BaseGate
from .blur_gate import BlurGate
from .clip_gate import ClipGate
from .entropy_gate import EntropyGate
from .grid_gate import GridGate
from .hash_gate import HashGate

# Import factory and convenience functions
from .factory import (
    GateFactory,
    register_all_gates,
    create_blur_gate,
    create_clip_gate,
    create_entropy_gate,
    create_grid_gate,
    create_hash_gate,
)

# Register all gates when the module is imported
register_all_gates()

# Define what gets exported when using "from cortexia_video.gates import *"
__all__ = [
    # Base class
    'BaseGate',
    
    # Gate implementations
    'BlurGate',
    'ClipGate', 
    'EntropyGate',
    'GridGate',
    'HashGate',
    
    # Factory
    'GateFactory',
    'register_all_gates',
    
    # Convenience functions
    'create_blur_gate',
    'create_clip_gate',
    'create_entropy_gate', 
    'create_grid_gate',
    'create_hash_gate',
]