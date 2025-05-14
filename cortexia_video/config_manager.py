import json
import logging
import os
from typing import Any, Dict, Optional

import toml
import yaml

from cortexia_video.object_listing import OBJECT_LISTER_REGISTRY


class ConfigManager:
    """Handles loading and accessing configuration from TOML, YAML or JSON files."""

    def __init__(
        self,
        config_file_path: Optional[str] = None,
        config_dir: str = "config",
        config_name: str = "config",
    ):
        """
        Initialize ConfigManager.

        Args:
            config_file_path: Direct path to a config file.
            config_dir: Directory containing config files (used if config_file_path is None).
            config_name: Base name of config file (without extension, used if config_file_path is None).
        """
        self.config_file_path = config_file_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.config_data: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> None:
        """Load configuration from the specified file path or search in the config directory."""
        if self.config_file_path and os.path.exists(self.config_file_path):
            file_path = self.config_file_path
            _, ext = os.path.splitext(file_path)
            if ext == ".toml":
                with open(file_path, "r") as f:
                    self.config_data = toml.load(f)
            elif ext in [".yml", ".yaml"]:
                with open(file_path, "r") as f:
                    self.config_data = yaml.safe_load(f)
            elif ext == ".json":
                with open(file_path, "r") as f:
                    self.config_data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file extension: {ext} for file {file_path}"
                )
            if not self.config_data:
                raise FileNotFoundError(
                    f"Config file found at {file_path} but is empty or invalid."
                )
            return

        # Fallback to searching in config_dir if config_file_path is not provided or not found
        toml_path = os.path.join(self.config_dir, f"{self.config_name}.toml")
        yaml_path = os.path.join(self.config_dir, f"{self.config_name}.yml")
        json_path = os.path.join(self.config_dir, f"{self.config_name}.json")

        if os.path.exists(toml_path):
            with open(toml_path, "r") as f:
                self.config_data = toml.load(f)
        elif os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                self.config_data = yaml.safe_load(f)
        elif os.path.exists(json_path):
            with open(json_path, "r") as f:
                self.config_data = json.load(f)
        else:
            paths_searched = [toml_path, yaml_path, json_path]
            if self.config_file_path:  # If a specific path was given but not found
                paths_searched.insert(0, self.config_file_path)
            raise FileNotFoundError(
                f"No config file found. Searched at: {', '.join(paths_searched)}"
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
        keys = key.split(".")
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
        model_name = self.get_param("model_settings.object_listing_model").lower()
        for pattern, lister_cls in OBJECT_LISTER_REGISTRY.items():
            if model_name.startswith(pattern):
                return lister_cls(self)
        raise ValueError(f"Unknown object listing model: {model_name}")

    def get_feature_extractor(self) -> Optional[Any]:
        """
        Retrieves an instance of the configured feature extractor.

        Note: This method requires clip_wrapper.py to be fully implemented.

        Returns:
            Optional[Any]: An instance of the feature extractor, or None if configuration is missing or invalid.
        """
        # Defer imports to avoid circular dependencies and allow for partial implementation
        try:
            # Only import when the method is called to avoid import errors at module load time
            from cortexia_video.clip_wrapper import (
                FEATURE_EXTRACTOR_REGISTRY,
                FeatureExtractor,
            )

            try:
                model_identifier = self.get_param(
                    "model_settings.clip_feature_model_identifier"
                )
                if not model_identifier:
                    self.logger.error(
                        "Feature extractor model identifier ('clip_feature_model_identifier') not found in config."
                    )
                    return None

                extractor_class = FEATURE_EXTRACTOR_REGISTRY.get(model_identifier)
                if extractor_class:
                    self.logger.info(f"Loading feature extractor: {model_identifier}")
                    return extractor_class(self)  # Pass self (ConfigManager instance)
                else:
                    self.logger.error(
                        f"No feature extractor found in registry for identifier: {model_identifier}"
                    )
                    return None
            except Exception as e:
                self.logger.error(
                    f"Error initializing feature extractor: {e}", exc_info=True
                )
                return None

        except ImportError as ie:
            self.logger.warning(f"Feature extractor module not available: {ie}")
            return None

    def set_param(self, key: str, value: Any) -> None:
        """
        Set a configuration parameter by dot notation key.

        Args:
            key: Dot notation key (e.g. 'logging.level')
            value: Value to set for the key
        """
        keys = key.split(".")
        current = self.config_data

        for k in keys:
            if k not in current:
                # add the key to the config data
                current[k] = {}
            current = current[k]

        current = value
