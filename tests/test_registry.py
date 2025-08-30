"""Tests for feature registry functionality"""

import pytest
from unittest.mock import Mock, patch

from cortexia.features.registry import feature_registry
from cortexia.features.base import BaseFeature
from cortexia.data.models.result.base_result import BaseResult


class MockRegistryFeature(BaseFeature):
    """Mock feature for registry testing"""
    
    output_schema = BaseResult
    required_inputs = []
    required_fields = []
    
    def _initialize(self):
        self.initialized = True
    
    def process_frame(self, frame, **inputs):
        return BaseResult()
    
    @property
    def name(self):
        return "mock_registry_feature"
    
    @property
    def description(self):
        return "Mock feature for registry testing"


class TestFeatureRegistry:
    """Test cases for feature registry"""
    
    def test_registry_initialization(self):
        """Test that registry is properly initialized"""
        assert feature_registry is not None
        assert feature_registry._name == "features"
    
    def test_register_feature(self):
        """Test registering a feature"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Register feature
        feature_registry.register("test_feature")(MockRegistryFeature)
        
        # Check registration
        assert "test_feature" in feature_registry._items
        assert feature_registry._items["test_feature"] == MockRegistryFeature
    
    def test_register_feature_with_decorator(self):
        """Test registering feature using decorator"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        @feature_registry.register("decorated_feature")
        class DecoratedFeature(BaseFeature):
            output_schema = BaseResult
            required_inputs = []
            required_fields = []
            
            def _initialize(self):
                self.initialized = True
            
            def process_frame(self, frame, **inputs):
                return BaseResult()
            
            @property
            def name(self):
                return "decorated_feature"
            
            @property
            def description(self):
                return "Decorated feature for testing"
        
        # Check registration
        assert "decorated_feature" in feature_registry._items
        assert feature_registry._items["decorated_feature"] == DecoratedFeature
    
    def test_get_feature(self):
        """Test getting a feature from registry"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Register feature
        feature_registry.register("get_test_feature")(MockRegistryFeature)
        
        # Get feature
        feature_class = feature_registry.get("get_test_feature")
        assert feature_class == MockRegistryFeature
    
    def test_get_nonexistent_feature(self):
        """Test getting a nonexistent feature"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Get nonexistent feature
        with pytest.raises(KeyError):
            feature_registry.require("nonexistent_feature")
    
    def test_list_features(self):
        """Test listing all registered features"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Register multiple features
        feature_registry.register("feature1")(MockRegistryFeature)
        feature_registry.register("feature2")(MockRegistryFeature)
        
        # List features
        features = list(feature_registry.keys())
        assert "feature1" in features
        assert "feature2" in features
        assert len(features) == 2
    
    def test_contains_feature(self):
        """Test checking if feature exists in registry"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Register feature
        feature_registry.register("contains_test_feature")(MockRegistryFeature)
        
        # Check contains
        assert "contains_test_feature" in feature_registry
        assert "nonexistent_feature" not in feature_registry
    
    def test_unregister_feature(self):
        """Test unregistering a feature"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Register feature
        feature_registry.register("unregister_test_feature")(MockRegistryFeature)
        
        # Check it exists
        assert "unregister_test_feature" in feature_registry
        
        # Unregister
        del feature_registry._items["unregister_test_feature"]
        
        # Check it's gone
        assert "unregister_test_feature" not in feature_registry
    
    def test_clear_registry(self):
        """Test clearing the registry"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Register multiple features
        feature_registry.register("clear_test_feature1")(MockRegistryFeature)
        feature_registry.register("clear_test_feature2")(MockRegistryFeature)
        
        # Check they exist
        assert len(feature_registry._items) == 2
        
        # Clear registry
        feature_registry._items.clear()
        
        # Check it's empty
        assert len(feature_registry._items) == 0
    
    def test_register_duplicate_feature(self):
        """Test registering a feature with duplicate name"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Register feature first time
        feature_registry.register("duplicate_feature")(MockRegistryFeature)
        
        # Register feature second time - should raise exception
        with pytest.raises(KeyError, match="already registered"):
            feature_registry.register("duplicate_feature")(MockRegistryFeature)
    
    def test_registry_repr(self):
        """Test registry string representation"""
        # Clear any existing registration
        feature_registry._items.clear()
        
        # Register a feature
        feature_registry.register("repr_test_feature")(MockRegistryFeature)
        
        # Check repr
        repr_str = repr(feature_registry)
        assert "Registry" in repr_str
        assert "features" in repr_str
        assert "1" in repr_str