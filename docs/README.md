# Cortexia Video SDK Documentation

Welcome to the official documentation for the Cortexia Video SDK. This comprehensive guide will help you understand, install, and use the modular computer vision framework for video annotation and analysis.

## 🚀 Quick Start

- **[Installation Guide](user_guide/installation.md)** - Get up and running quickly
- **[Getting Started](user_guide/getting_started.md)** - Your first steps with Cortexia
- **[Basic Usage](user_guide/basic_usage.md)** - Common usage patterns and examples

## 📚 Documentation Structure

### User Guide
For users who want to use the Cortexia Video SDK in their projects.

| Document | Description |
|----------|-------------|
| [Getting Started](user_guide/getting_started.md) | Quick start guide and first steps |
| [Installation](user_guide/installation.md) | Detailed installation instructions |
| [Configuration](user_guide/configuration.md) | Configuration options and examples |
| [Basic Usage](user_guide/basic_usage.md) | Basic usage patterns and workflows |
| [Advanced Usage](user_guide/advanced_usage.md) | Advanced features and patterns |
| [Migration Guide](user_guide/migration_guide.md) | Migrating from previous versions |
| [Troubleshooting](user_guide/troubleshooting.md) | Common issues and solutions |

### Architecture
For developers who want to understand the technical architecture and design principles.

| Document | Description |
|----------|-------------|
| [Overview](architecture/overview.md) | High-level architecture overview |
| [Modular Design](architecture/modular_design.md) | Modular architecture principles |
| [Data Flow](architecture/data_flow.md) | Data flow and pipeline architecture |
| [Component Architecture](architecture/component_architecture.md) | Detailed component architecture |
| [Registry System](architecture/registry_system.md) | Registry system and extensibility |
| [Performance Considerations](architecture/performance_considerations.md) | Performance characteristics |

### API Reference
For developers who need detailed API documentation.

| Document | Description |
|----------|-------------|
| [Overview](api_reference/overview.md) | API overview and conventions |
| [Core API](api_reference/core_api.md) | Core SDK interfaces |
| [Features](api_reference/features.md) | Feature components API |
| [Gates](api_reference/gates.md) | Gate components API |
| [Samplers](api_reference/samplers.md) | Sampler components API |
| [Data Models](api_reference/data_models.md) | Data models and schemas |
| [Configuration API](api_reference/configuration_api.md) | Configuration API |

### Developer Guide
For contributors who want to develop and extend the Cortexia Video SDK.

| Document | Description |
|----------|-------------|
| [Getting Started](developer_guide/getting_started.md) | Setting up development environment |
| [Coding Standards](developer_guide/coding_standards.md) | Coding standards and conventions |
| [Testing Guide](developer_guide/testing_guide.md) | Testing framework and practices |
| [Contributing](developer_guide/contributing.md) | Contribution guidelines |
| [Architecture Decisions](developer_guide/architecture_decisions.md) | Architecture decision records |
| [Release Process](developer_guide/release_process.md) | Release process and versioning |

### Deployment
For operations teams who need to deploy and manage Cortexia Video SDK.

| Document | Description |
|----------|-------------|
| [Docker](deployment/docker.md) | Docker deployment guide |
| [Scaling](deployment/scaling.md) | Scaling and performance optimization |
| [Monitoring](deployment/monitoring.md) | Monitoring and observability |
| [Cloud Deployment](deployment/cloud_deployment.md) | Cloud deployment options |

### Examples
Practical examples and tutorials for common use cases.

| Document | Description |
|----------|-------------|
| [Basic Examples](examples/basic_examples.md) | Basic usage examples |
| [Advanced Examples](examples/advanced_examples.md) | Advanced usage examples |
| [Integration Examples](examples/integration_examples.md) | Integration with other tools |
| [Performance Examples](examples/performance_examples.md) | Performance optimization examples |

## 🎯 Key Concepts

### Modular Architecture
Cortexia Video SDK is built on a modular architecture that allows you to compose custom processing workflows using independent components:

- **Features**: Extract annotations and analysis from video frames
- **Gates**: Apply quality filters and criteria to frames
- **Samplers**: Select frames intelligently from video streams

### Core Benefits
- **Modular Design**: Use only the components you need
- **Flexible Composition**: Combine components in any order
- **Registry System**: Easy to extend with custom implementations
- **Unified Data Management**: Consistent interfaces across all components
- **Batch Processing**: Efficient handling of large datasets

## 🛠️ Quick Example

```python
import cortexia

# Initialize the SDK
cortexia_sdk = cortexia.Cortexia()

# Create components
detector = cortexia.create_feature("detection")
blur_gate = cortexia.create_gate("blur")

# Process a frame
frame = load_video_frame("video.mp4", frame_number=0)
if blur_gate.process_frame(frame).passed:
    result = detector.process_frame(frame)
    print(f"Detected {len(result.detections)} objects")
```

## 📖 Learning Path

### For New Users
1. Read the [Installation Guide](user_guide/installation.md)
2. Follow the [Getting Started](user_guide/getting_started.md) tutorial
3. Explore [Basic Usage](user_guide/basic_usage.md) patterns
4. Check out the [Examples](examples/) for practical implementations

### For Advanced Users
1. Review the [Architecture](architecture/) documentation
2. Explore the [API Reference](api_reference/) for detailed information
3. Read the [Advanced Usage](user_guide/advanced_usage.md) guide
4. Learn about [Performance Optimization](deployment/scaling.md)

### For Contributors
1. Set up your development environment with the [Developer Guide](developer_guide/getting_started.md)
2. Follow the [Coding Standards](developer_guide/coding_standards.md)
3. Understand the [Testing Framework](developer_guide/testing_guide.md)
4. Review the [Contribution Guidelines](developer_guide/contributing.md)

## 🆘 Getting Help

- **Documentation**: Browse this documentation site
- **Examples**: Check the [cookbook/](../cookbook/) directory for practical examples
- **Issues**: Report bugs or request features on GitHub
- **Community**: Join discussions and ask questions

## 📝 Feedback

We welcome feedback on the documentation! If you find any issues, have suggestions for improvements, or want to contribute, please:

1. Check the [Contributing](developer_guide/contributing.md) guide
2. Open an issue on GitHub
3. Submit a pull request with your improvements

---

**Last Updated**: August 2025  
**Version**: 0.1.0  
**License**: See project license file