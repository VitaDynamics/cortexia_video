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

    def validate_batch_size(self) -> None:
        """
        Validate that the batch_size parameter is an integer if it exists.

        Raises:
            ValueError: If batch_size is not an integer
        """
        if (
            "processing" in self.config_data
            and "batch_size" in self.config_data["processing"]
        ):
            batch_size = self.config_data["processing"]["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError(
                    f"batch_size must be a positive integer, got {batch_size} ({type(batch_size).__name__})"
                )
            self.logger.info(f"Validated batch_size: {batch_size}")

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
            # Validate batch_size if present
            self.validate_batch_size()
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

        # Validate batch_size if config was loaded successfully
        if self.config_data:
            self.validate_batch_size()
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
        current = self.config_data

        try:
            for k in keys:
                if not isinstance(current, dict) or k not in current:
                    return default
                current = current[k]
            return current
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Config parameter '{key}' not found")

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
            from cortexia_video.feature_extraction import (
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

        # Navigate to the last parent dictionary
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                # add the key to the config data
                current[k] = {}
            current = current[k]

        # Set the value at the last key
        current[keys[-1]] = value
