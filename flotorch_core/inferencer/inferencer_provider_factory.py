

from typing import Dict, List
from flotorch_core.inferencer.bedrock_inferencer import BedrockInferencer
from flotorch_core.inferencer.gateway_inferencer import GatewayInferencer
from flotorch_core.inferencer.inferencer import BaseInferencer
from flotorch_core.inferencer.sagemaker_inferencer import SageMakerInferencer
from flotorch_core.inferencer.llama_inferencer import LlamaInferencer


class InferencerProviderFactory:
    """
    Factory to create inferencer based on the service name.
    """
    @staticmethod
    def create_inferencer_provider(gateway_enabled: bool, base_url: str, api_key: str, service: str, model_id: str, region: str, arn_role: str, n_shot_prompts: int = 0, temperature: float = 0.7, n_shot_prompt_guide_obj: Dict[str, List[Dict[str, str]]] = None) -> BaseInferencer:
        """
        Create an inferencer provider based on the service name.
        Args:
            gateway_enabled (bool): Whether to use the gateway.
            base_url (str): The base URL for the gateway.
            api_key (str): The API key for the gateway.
            service (str): The service name (e.g., 'bedrock', 'sagemaker').
            model_id (str): The model ID.
            region (str): The AWS region.
            arn_role (str): The ARN role for SageMaker.
            n_shot_prompts (int): Number of shot prompts.
            temperature (float): Temperature for sampling.
            n_shot_prompt_guide_obj (Dict[str, List[Dict[str, str]]]): N-shot prompt guide object.
        Returns:
            BaseInferencer: An instance of the appropriate inferencer.
        """
        if gateway_enabled:
            return GatewayInferencer(
                model_id=model_id, 
                api_key=api_key, 
                base_url=base_url, 
                n_shot_prompts=n_shot_prompts, 
                n_shot_prompt_guide_obj=n_shot_prompt_guide_obj
            )
        
        if service == 'bedrock':
            return BedrockInferencer(model_id, region, n_shot_prompts, temperature, n_shot_prompt_guide_obj)
        elif service == 'sagemaker':
            if model_id.startswith("meta-vlm-llama-4"):
                return LlamaInferencer(model_id, region, arn_role, n_shot_prompts, temperature, n_shot_prompt_guide_obj)
            else:
                return SageMakerInferencer(model_id, region, arn_role, n_shot_prompts, temperature, n_shot_prompt_guide_obj)
        else:
            raise ValueError(f"Unsupported service scheme: {service}")
