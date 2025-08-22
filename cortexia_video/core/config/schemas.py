"""Configuration schemas for validation"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    """Configuration for individual features"""
    
    model: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)


class ProcessingConfig(BaseModel):
    """Configuration for processing settings"""
    
    batch_size: int = 8
    device: str = "auto"
    max_workers: int = 4
    memory_limit: Optional[str] = None
    gpu_memory_fraction: float = 0.8


class LoggingConfig(BaseModel):
    """Configuration for logging settings"""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5


class OutputConfig(BaseModel):
    """Configuration for output settings"""
    
    save_intermediate: bool = False
    save_final: bool = True
    format: str = "json"
    include_images: bool = False
    compression: bool = True


class CortexiaConfig(BaseModel):
    """Main configuration schema for Cortexia"""
    
    features: Dict[str, FeatureConfig] = Field(default_factory=dict)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
    
    def get_feature_config(self, feature_name: str) -> Optional[FeatureConfig]:
        """Get configuration for a specific feature"""
        return self.features.get(feature_name)
    
    def set_feature_config(self, feature_name: str, config: FeatureConfig):
        """Set configuration for a specific feature"""
        self.features[feature_name] = config
    
    def enable_feature(self, feature_name: str):
        """Enable a specific feature"""
        if feature_name in self.features:
            self.features[feature_name].enabled = True
    
    def disable_feature(self, feature_name: str):
        """Disable a specific feature"""
        if feature_name in self.features:
            self.features[feature_name].enabled = False
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.features.get(feature_name, FeatureConfig(model="")).enabled
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled features"""
        return [
            name for name, config in self.features.items()
            if config.enabled
        ]
    
    def merge(self, other: "CortexiaConfig") -> "CortexiaConfig":
        """Merge with another configuration"""
        merged_features = self.features.copy()
        merged_features.update(other.features)
        
        return CortexiaConfig(
            features=merged_features,
            processing=other.processing,
            logging=other.logging,
            output=other.output,
            metadata={**self.metadata, **other.metadata}
        )


# Default configurations
DEFAULT_FEATURE_CONFIGS = {
    "detection": FeatureConfig(
        model="IDEA-Research/grounding-dino-base",
        config={
            "box_threshold": 0.3,
            "text_threshold": 0.3,
            "default_prompts": ["object"]
        }
    ),
    "listing": FeatureConfig(
        model="vikhyatk/moondream2",
        config={
            "task_prompt": "List all objects in this image."
        }
    ),
    "segmentation": FeatureConfig(
        model="facebook/sam-vit-huge",
        config={
            "points_per_side": 32,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "crop_n_layers": 0,
            "min_mask_region_area": 100.0
        }
    ),
    "description": FeatureConfig(
        model="Salesforce/blip2-opt-2.7b",
        config={
            "prompt": "Describe this object in detail."
        }
    ),
    "feature_extraction": FeatureConfig(
        model="openai/clip-vit-base-patch32",
        config={}
    ),
    "caption": FeatureConfig(
        model="vikhyatk/moondream2",
        config={
            "caption_length": "long",
            "prompt": "Generate a detailed caption for this image:"
        }
    )
}


def create_default_config() -> CortexiaConfig:
    """Create default configuration with all features"""
    return CortexiaConfig(
        features=DEFAULT_FEATURE_CONFIGS.copy(),
        processing=ProcessingConfig(),
        logging=LoggingConfig(),
        output=OutputConfig()
    )


def create_minimal_config() -> CortexiaConfig:
    """Create minimal configuration with essential features"""
    minimal_features = {
        "detection": DEFAULT_FEATURE_CONFIGS["detection"],
        "listing": DEFAULT_FEATURE_CONFIGS["listing"],
    }
    
    return CortexiaConfig(
        features=minimal_features,
        processing=ProcessingConfig(),
        logging=LoggingConfig(),
        output=OutputConfig()
    )