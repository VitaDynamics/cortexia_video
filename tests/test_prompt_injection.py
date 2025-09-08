import unittest
from unittest.mock import patch

from cortexia.features.listing.lister import ListingFeature


class TestPromptInjection(unittest.TestCase):
    @patch("cortexia.features.listing.lister.Qwen2_5VLLister")
    def test_prompt_injection(self, MockedQwenLister):
        """
        Tests that the prompt is correctly propagated to the lister.
        """
        # 1. Test default prompt from decorator
        feature_with_default_prompt = ListingFeature()
        feature_with_default_prompt._initialize()

        # The first call to the constructor should be with the default prompt.
        lister_config = MockedQwenLister.call_args[0][0]
        self.assertEqual(
            lister_config["task_prompt"], "List all objects in this image."
        )

        # 2. Test custom prompt passed to constructor
        custom_prompt = "Describe the scene in detail."
        feature_with_custom_prompt = ListingFeature(prompt=custom_prompt)
        feature_with_custom_prompt._initialize()

        # The second call to the constructor should be with the custom prompt.
        lister_config = MockedQwenLister.call_args[0][0]
        self.assertEqual(lister_config["task_prompt"], custom_prompt)


if __name__ == "__main__":
    unittest.main()
