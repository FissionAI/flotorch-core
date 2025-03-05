from inferencer.inferencer import BaseInferencer
from typing import List, Dict, Any, Tuple
from logger.global_logger import get_logger
import boto3
import random


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

    # TODO: retry needs to be implemented
    def generate_text(self, user_query: str, context: List[Dict]) -> Tuple[Dict[Any, Any], str]:
        """
        Generate a response based on the user query and context using Bedrock.

        Args:
            user_query (str): The question or input from the user.
            context (List[Dict]): Contextual data to assist in generating a response.

        Returns:
            Tuple[Dict[Any, Any], str]: Metadata and the generated response text.
        """
        try:
            converse_prompt = self.generate_prompt(user_query, context)
            messages = self._prepare_payload(converse_prompt)
            inference_config={"maxTokens": 512, "temperature": self.temperature, "topP": 0.9}
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig=inference_config
            )
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

    def generate_prompt(self, user_query: str, context: List[Dict]) -> str:
        """
        Construct a prompt for the Bedrock inferencer based on the user query and context.

        Args:
            user_query (str): The question or input from the user.
            context (List[Dict]): Contextual data to assist in generating a response.

        Returns:
            str: The constructed prompt.
        """
        system_prompt = self.n_shot_prompt_guide_obj.get("system_prompt")

        context_text = self.format_context(context)

        base_prompt = self.n_shot_prompt_guide_obj.get("user_prompt", "") if self.n_shot_prompt_guide_obj else ""

        if self.n_shot_prompts == 0:
            prompt = (
                f"{system_prompt}\n\n"
                f"<context>\n{context_text}\n</context>\n"
                f"{base_prompt}\n"
                f"Question: {user_query}"
            )
            return prompt.strip()

        examples = self.n_shot_prompt_guide_obj.get("examples", []) if self.n_shot_prompt_guide_obj else []
        selected_examples = (
            random.sample(examples, self.n_shot_prompts)
            if len(examples) > self.n_shot_prompts
            else examples
        )

        example_text = "\n".join([f"- {example['example']}" for example in selected_examples])

        prompt = (
            f"{system_prompt}\n\n"
            f"Few examples:\n{example_text}\n\n"
            f"<context>\n{context_text}\n</context>\n"
            f"{base_prompt}\n"
            f"Question: {user_query}"
        )

        return prompt.strip()

    def _prepare_payload(self, prompt: str):
        conversation = [
            {
                "role": "user", 
                "content": [{"text" : f"{prompt}"}]
            }
        ]
        return conversation

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
