"""
Cortexia Video Gates Module

This module provides various gate implementations for filtering video frames
based on different criteria such as blur, entropy, content similarity, etc.

Gates are filters that analyze video frames and determine whether they should
pass through (return True) or be filtered out (return False).

Example usage:

    # Using individual gate classes
    from cortexia.gates import BlurGate
    blur_gate = BlurGate(threshold=100.0)
    
    # Using the gate hub
    from cortexia.core.typed_registry import gate_hub
    blur_gate = gate_hub.create('blur', threshold=100.0)
    
    # Using convenience functions
    from cortexia.gates import create_blur_gate
    blur_gate = create_blur_gate(threshold=100.0)
"""

# Import all gate classes
from .base_gate import BaseGate
from .blur_gate import BlurGate
from .clip_gate import ClipGate
from .entropy_gate import EntropyGate
from .grid_gate import GridGate
from .hash_gate import HashGate
from .registry import gate_registry

# Define what gets exported when using "from cortexia.gates import *"
__all__ = [
    "gate_registry",
    'BaseGate',
    'ClipGate', 
    'EntropyGate',
    'GridGate',
    'HashGate',
]