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