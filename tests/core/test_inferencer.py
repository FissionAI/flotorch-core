import unittest
from unittest.mock import Mock, patch

from flotorch_core.inferencer.inferencer_provider_factory import InferencerProviderFactory
from flotorch_core.inferencer.gateway_inferencer import GatewayInferencer
from flotorch_core.inferencer.inferencer import DEFAULT_SYSTEM_PROMPT


class TestGatewayInferencerFactory(unittest.TestCase):
    """Test GatewayInferencer creation through InferencerProviderFactory"""
    
    def test_create_gateway_inferencer_through_factory(self):
        """Test creating GatewayInferencer using the factory with gateway_enabled=True"""
        inferencer = InferencerProviderFactory.create_inferencer_provider(
            gateway_enabled=True,
            base_url="",
            api_key="",
            service="",
            model_id="flotorch/haiku-long",
            region="",
            arn_role=""
        )
        
        metadata, response = inferencer.generate_text("What is the capital of France?", [])

        self.assertIsInstance(inferencer, GatewayInferencer)
        self.assertEqual(inferencer.model_id, "flotorch/haiku-long")
        self.assertEqual(inferencer.api_key, "")
        self.assertEqual(inferencer.base_url, "")
        self.assertIsNone(inferencer.n_shot_prompt_guide_obj)
        self.assertIsNotNone(response)
        self.assertIsNotNone(metadata)


if __name__ == '__main__':
    unittest.main()
