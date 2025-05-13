import os
import json
import yaml
import toml
from typing import Any, Dict, Optional
from cortexia_video.object_listing import OBJECT_LISTER_REGISTRY

class ConfigManager:
    """Handles loading and accessing configuration from TOML, YAML or JSON files."""
    
    def __init__(self, config_dir: str = "config", config_name: str = "config"):
        """
        Initialize ConfigManager with directory and base config name.
        
        Args:
            config_dir: Directory containing config files
            config_name: Base name of config file (without extension)
        """
        self.config_dir = config_dir
        self.config_name = config_name
        self.config_data: Dict[str, Any] = {}
        
    def load_config(self) -> None:
        """Load configuration from TOML, YAML or JSON file (preferring in that order)."""
        toml_path = os.path.join(self.config_dir, f"{self.config_name}.toml")
        yaml_path = os.path.join(self.config_dir, f"{self.config_name}.yml")
        json_path = os.path.join(self.config_dir, f"{self.config_name}.json")
        
        if os.path.exists(toml_path):
            with open(toml_path, 'r') as f:
                self.config_data = toml.load(f)
        elif os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                self.config_data = yaml.safe_load(f)
        elif os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.config_data = json.load(f)
        else:
            raise FileNotFoundError(
                f"No config file found at {toml_path}, {yaml_path}, or {json_path}"
            )
            
    def get_param(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get configuration parameter by dot notation key.
        
        Args:
            key: Dot notation key (e.g. 'logging.level')
            default: Default value if key not found
            
        Returns:
            The configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if k not in value:
                if default is not None:
                    return default
                raise KeyError(f"Config parameter '{key}' not found")
            value = value[k]
        return value
            
    def validate_config(self, required_keys: list[str]) -> bool:
        """
        Validate that required configuration keys are present.
        
        Args:
            required_keys: List of required keys in dot notation
            
        Returns:
            True if all keys are present, False otherwise
        """
        missing_keys = []
        for key in required_keys:
            try:
                self.get_param(key)
            except KeyError:
                missing_keys.append(key)
                
        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing_keys)}"
            )
        return True

    def get_object_lister(self):
        """
        Return the correct ObjectLister instance based on the config value using the registry pattern.
        """
        model_name = self.get_param('model_settings.object_listing_model').lower()
        for pattern, lister_cls in OBJECT_LISTER_REGISTRY.items():
            if model_name.startswith(pattern):
                return lister_cls(self)
        raise ValueError(f"Unknown object listing model: {model_name}")