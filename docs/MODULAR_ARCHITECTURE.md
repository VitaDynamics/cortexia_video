# Cortexia Video SDK - Modular Architecture Design

## Core Philosophy

**Independent Components, Unified Data Management**

Each feature, gate, and sampler is an independent SDK component. Data source and storage are unified into a single Data Manager that handles both input and output.

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CORTEXIA VIDEO SDK                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Annotation     │  │  Gates          │  │  Samplers   │ │
│  │  Features       │  │  (Filters)      │  │  (Selection)│ │
│  │                 │  │                 │  │             │ │
│  │ • Detection     │  │ • Resolution    │  │ • Random    │ │
│  │ • Segmentation  │  │ • Content       │  │ • Stratified│ │
│  │ • Caption       │  │ • Metadata      │  │ • Temporal  │ │
│  │ • Depth         │  │ • Custom        │  │ • Quality   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Data Manager Layer                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 DataManager                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │  │  Connectors │  │  Storage    │  │  Metadata       │ │ │
│  │  │  (Input)    │  │  (Output)   │  │  Manager        │ │ │
│  │  │             │  │             │  │                 │ │ │
│  │  │ • Local     │  │ • Local     │  │ • Indexing     │ │ │
│  │  │ • Remote    │  │ • Database  │  │ • Search       │ │ │
│  │  │ • Database  │  │ • Cloud     │  │ • Versioning   │ │ │
│  │  │ • Cloud     │  │ • VectorDB  │  │ • Lineage      │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```


## Benefits of This Design

1. **Modularity**: Each component is completely independent
2. **Composability**: Components can be combined in any order
3. **Simplicity**: No complex pipeline orchestration
4. **Testability**: Each component can be tested independently
5. **Extensibility**: Easy to add new gates, samplers, or features
6. **Flexibility**: Users can pick and choose components they need
7. **Unified Data Management**: Single interface for input/output

This design gives you a true SDK where users can compose their own processing flows using independent components, while maintaining clean data management through unified DataManager.

## Universal Registry & Decorators

- Purpose: Provide a simple, explicit way to register implementations for features, gates, and inner model choices (e.g., captioners, listers) using decorators.
- Location: `cortexia_video/core/registry.py`
- API: `Registry`, `create_registry(kind)` with `register`, `decorator`, `get`, `create`, `items`, `keys`.

Usage patterns:

- Features and gates: Use factory decorators to auto-register classes on import.

  - Feature: `@FeatureFactory.decorator("my_feature", default_config={...})`
  - Gate: `@GateFactory.decorator("my_gate", default_config={...})`

- Inner registries (e.g., captioners, listers): Create a registry and decorate implementations.

  - Define registry: `MY_REG = create_registry("my_kind")`
  - Decorate: `@MY_REG.decorator("impl_name", aliases=["alias"])`

Example (captioners):

```
from cortexia_video.core.registry import create_registry

IMAGE_CAPTIONER_REGISTRY = create_registry("image_captioner")

@IMAGE_CAPTIONER_REGISTRY.decorator("vikhyatk/moondream2", aliases=["moondream2"])
class MoonDreamCaptioner(ImageCaptioner):
    ...

# Lookup in feature code
captioner_cls = IMAGE_CAPTIONER_REGISTRY.get(model_name)
captioner = captioner_cls(config)
```

Notes:

- Registration happens at import time. Ensure modules are imported (e.g., via existing `register_all_features()` or explicit imports) so decorators run.
- Keys are case-insensitive; aliases allow multiple names to map to one implementation.
- The registry is minimal by design — no hidden scanning or magic, just clear and explicit hooks.
