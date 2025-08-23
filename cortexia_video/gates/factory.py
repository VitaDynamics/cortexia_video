"""
Gate Factory Registry for creating and managing different gate implementations.

This factory provides a centralized way to register and create gate instances
with appropriate initialization parameters.
"""
from typing import Dict, Type, Any, Optional, Callable
from .base_gate import BaseGate


class GateFactory:
    """
    Factory class for creating gate instances with a registry pattern.
    
    Provides a clean interface for registering gate classes and creating
    instances with appropriate parameters.
    """
    
    _registry: Dict[str, Type[BaseGate]] = {}
    _default_configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, gate_class: Type[BaseGate], default_config: Optional[Dict[str, Any]] = None):
        """
        Register a gate class with the factory.
        
        Args:
            name: Name to register the gate under (e.g., 'blur', 'entropy')
            gate_class: The gate class to register
            default_config: Optional default configuration parameters
        """
        cls._registry[name.lower()] = gate_class
        cls._default_configs[name.lower()] = default_config or {}
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseGate:
        """
        Create a gate instance by name.
        
        Args:
            name: Name of the gate to create
            **kwargs: Additional parameters to pass to the gate constructor
            
        Returns:
            BaseGate: Instance of the requested gate
            
        Raises:
            ValueError: If the gate name is not registered
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Gate '{name}' not found. Available gates: {list(cls._registry.keys())}")
        
        gate_class = cls._registry[name]
        default_config = cls._default_configs[name].copy()
        
        # Merge default config with provided kwargs
        config = {**default_config, **kwargs}
        
        return gate_class(**config)
    
    @classmethod
    def get_available_gates(cls) -> list[str]:
        """
        Get list of available gate names.
        
        Returns:
            List of registered gate names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_gate_class(cls, name: str) -> Type[BaseGate]:
        """
        Get the gate class by name without instantiating.
        
        Args:
            name: Name of the gate
            
        Returns:
            Gate class
            
        Raises:
            ValueError: If the gate name is not registered
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Gate '{name}' not found. Available gates: {list(cls._registry.keys())}")
        
        return cls._registry[name]
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered gates. Primarily for testing."""
        cls._registry.clear()
        cls._default_configs.clear()

    # Decorator-based registration for convenience
    @classmethod
    def decorator(cls, name: str, default_config: Optional[Dict[str, Any]] = None) -> Callable[[Type[BaseGate]], Type[BaseGate]]:
        """Return a decorator that registers a gate class under `name`."""
        def _decorator(gate_class: Type[BaseGate]) -> Type[BaseGate]:
            cls.register(name, gate_class, default_config)
            return gate_class
        return _decorator


def register_all_gates():
    """
    Register all available gate implementations with the factory.
    This function should be called to populate the factory registry.
    """
    # Import gates locally to avoid circular imports
    from .blur_gate import BlurGate
    from .clip_gate import ClipGate
    from .entropy_gate import EntropyGate
    from .grid_gate import GridGate
    from .hash_gate import HashGate
    
    # Register each gate with appropriate default configs
    GateFactory.register('blur', BlurGate, {
        'threshold': BlurGate.DEFAULT_THRESHOLD,
        'session_id': 'default_session'
    })
    
    GateFactory.register('clip', ClipGate, {
        'similarity_threshold': 0.8,
        'session_id': 'default_session',
        'clip_model_name': 'ViT-B/32',
        'clip_pretrained_weights': 'openai'
    })
    
    GateFactory.register('entropy', EntropyGate, {
        'threshold': EntropyGate.DEFAULT_THRESHOLD,
        'session_id': 'default_session'
    })
    
    GateFactory.register('grid', GridGate, {
        'grid_rows': GridGate.DEFAULT_GRID_ROWS,
        'grid_cols': GridGate.DEFAULT_GRID_COLS,
        'cell_hash_size': GridGate.DEFAULT_CELL_HASH_SIZE,
        'similarity_threshold': GridGate.DEFAULT_SIMILARITY_THRESHOLD,
        'session_id': 'default_session'
    })
    
    GateFactory.register('hash', HashGate, {
        'hash_type': HashGate.DEFAULT_HASH_TYPE,
        'hash_size': HashGate.DEFAULT_HASH_SIZE,
        'highfreq_factor': HashGate.DEFAULT_HIGHFREQ_FACTOR,
        'threshold': HashGate.DEFAULT_THRESHOLD,
        'session_id': 'default_session'
    })


# Convenience functions for easy gate creation
def create_blur_gate(threshold: Optional[float] = None, session_id: str = "default_session") -> BaseGate:
    """Create a blur gate with specified parameters."""
    kwargs = {'session_id': session_id}
    if threshold is not None:
        kwargs['threshold'] = threshold
    return GateFactory.create('blur', **kwargs)


def create_clip_gate(
    similarity_threshold: float = 0.8,
    session_id: str = "default_session",
    clip_model_name: str = "ViT-B/32",
    clip_pretrained_weights: str = "openai",
    device: Optional[str] = None
) -> BaseGate:
    """Create a CLIP gate with specified parameters."""
    kwargs = {
        'similarity_threshold': similarity_threshold,
        'session_id': session_id,
        'clip_model_name': clip_model_name,
        'clip_pretrained_weights': clip_pretrained_weights
    }
    if device is not None:
        kwargs['device'] = device
    return GateFactory.create('clip', **kwargs)


def create_entropy_gate(threshold: Optional[float] = None, session_id: str = "default_session") -> BaseGate:
    """Create an entropy gate with specified parameters."""
    kwargs = {'session_id': session_id}
    if threshold is not None:
        kwargs['threshold'] = threshold
    return GateFactory.create('entropy', **kwargs)


def create_grid_gate(
    grid_rows: int = 4,
    grid_cols: int = 4,
    cell_hash_size: int = 8,
    similarity_threshold: int = 5,
    session_id: str = "default_session"
) -> BaseGate:
    """Create a grid gate with specified parameters."""
    return GateFactory.create('grid', 
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        cell_hash_size=cell_hash_size,
        similarity_threshold=similarity_threshold,
        session_id=session_id
    )


def create_hash_gate(
    hash_type: str = "phash",
    hash_size: int = 8,
    highfreq_factor: int = 4,
    threshold: float = 0.05,
    session_id: str = "default_session"
) -> BaseGate:
    """Create a hash gate with specified parameters."""
    return GateFactory.create('hash',
        hash_type=hash_type,
        hash_size=hash_size,
        highfreq_factor=highfreq_factor,
        threshold=threshold,
        session_id=session_id
    )
