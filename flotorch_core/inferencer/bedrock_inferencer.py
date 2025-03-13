from flotorch_core.inferencer.inferencer import BaseInferencer
from typing import List, Dict, Any, Tuple
from flotorch_core.logger.global_logger import get_logger
import boto3
import random

from flotorch_core.utils.bedrock_retry_handler import BedRockRetryHander


logger = get_logger()

class BedrockInferencer(BaseInferencer):
    """
    Bedrock-specific implementation of the BaseInferencer.
    """

    def __init__(self, model_id: str, region: str = "us-east-1", n_shot_prompts: int = 0, temperature: float = 0.7, n_shot_prompt_guide_obj: Dict[str, List[Dict[str, str]]] = None):
        """
        Initialize the BedrockInferencer with Bedrock-specific parameters.

        Args:
            model_id (str): Identifier for the Bedrock model.
            region (str): AWS region where the Bedrock service is deployed.
            n_shot_prompts (int): Number of examples to include in few-shot learning.
            temperature (float): Sampling temperature for response generation.
            n_shot_prompt_guide_obj (Dict[str, List[Dict[str, str]]]): Guide object for few-shot examples.
        """
        super().__init__(model_id, region, n_shot_prompts, temperature, n_shot_prompt_guide_obj)
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )

    @BedRockRetryHander()
    def generate_text(self, user_query: str, context: List[Dict] = None) -> Tuple[Dict[Any, Any], str]:
        """
        Generate a response based on the user query and context using Bedrock.
        """
        try:
            system_prompt, messages = self.generate_prompt(user_query, context)
            
            inference_config = {
                "maxTokens": 512, 
                "temperature": self.temperature, 
                "topP": 0.9
            }
            
            skip_system_param = self.model_id in ("amazon.titan-text-express-v1", "amazon.titan-text-lite-v1", "mistral.mistral-7b-instruct-v0:2")
            request_params = {
                "modelId": self.model_id,
                "messages": ([self._prepare_conversation(role="user", message=system_prompt)] if skip_system_param else []) + messages,
                "inferenceConfig": inference_config
            }
            
            if not skip_system_param:
                request_params["system"] = [{"text": system_prompt}]
            
            response = self.client.converse(**request_params)
            
            metadata = {}
            if 'usage' in response:
                for key, value in response['usage'].items():
                    metadata[key] = value
            if 'metrics' in response:
                for key, value in response['metrics'].items():
                    metadata[key] = value
            
            return metadata, self._extract_response(response)
        except Exception as e:
            logger.error(f"Error generating text with Bedrock: {str(e)}")
            raise

    def generate_prompt(self, user_query: str, context: List[Dict] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Construct a prompt for the Bedrock inferencer based on the user query and context.
        """
        messages = []
        context_text = ""
        
        # Validate n_shot_prompt
        if self.n_shot_prompts < 0:
            raise ValueError("n_shot_prompts must be non-negative")
        
        default_prompt = "You are a helpful assistant. Use the provided context to answer questions accurately. If you cannot find the answer in the context, say so"
        # Get system prompt
        system_prompt = default_prompt if not self.n_shot_prompt_guide_obj or not self.n_shot_prompt_guide_obj.get("system_prompt") else self.n_shot_prompt_guide_obj.get("system_prompt")
        
        # Process context
        if context:
            context_text = self.format_context(context)
            if context_text:
                messages.append(self._prepare_conversation(role="user", message=context_text))
        
        base_prompt = self.n_shot_prompt_guide_obj.get("user_prompt", "") if self.n_shot_prompt_guide_obj else ""
        if base_prompt:
            messages.append(self._prepare_conversation(role="user", message=base_prompt))
        
        # Get examples
        examples = self.n_shot_prompt_guide_obj.get("examples", []) if self.n_shot_prompt_guide_obj else []
        selected_examples = random.sample(examples, self.n_shot_prompts) if len(examples) > self.n_shot_prompts else examples
        
        # Format examples
        for example in selected_examples:
            if 'example' in example:
                messages.append(self._prepare_conversation(role="user", message=example['example']))
            elif 'question' in example and 'answer' in example:
                messages.append(self._prepare_conversation(role="user", message=example['question']))
                messages.append(self._prepare_conversation(role="assistant", message=example['answer']))
        
        logger.info(f"Using {self.n_shot_prompts} shot prompt with {len(selected_examples)} examples")
        
        # Add user query
        messages.append(self._prepare_conversation(role="user", message=user_query))
        
        return system_prompt, messages

    def _prepare_conversation(self, message: str, role: str) -> Dict[str, Any]:
        """Formats a message and role into a conversation dictionary."""
        if not message or not role:
            logger.error("Error in parsing message or role")
        
        return {"role": role, "content": [{"text": message}]}

    def format_context(self, context: List[Dict[str, str]]) -> str:
        """Format context documents into a single string."""
        if not context or len(context) == 0:
            return ""
        
        context_text = "\n".join([
            f"Context {i+1}:\n{doc.get('text', '')}"
            for i, doc in enumerate(context)
        ])
        logger.debug(f"Formatted context text length: {len(context_text)}")
        return context_text

    def _extract_response(self, response: Dict) -> str:
        """
        Extract the generated text from the Bedrock response.

        Args:
            response (Dict): Response from the Bedrock API.

        Returns:
            str: Extracted text from the response.
        """
        response_text = response["output"]["message"]["content"][0]["text"]
        logger.debug(f"Response length: {len(response_text)}")
        return response_text
