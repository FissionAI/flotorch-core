from typing import List, Dict, Tuple, Any
import logging
import time
import random
from .sagemaker_inferencer import SageMakerInferencer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LlamaInferencer(SageMakerInferencer):
    """
    LlamaInferencer is a class that handles the inference process for the Llama-4 model.
    It extends the SageMakerInferencer class and provides methods to prepare prompts, construct payloads,
    and extract responses from the model.
    """
    def __init__(self, model_id: str, region: str, role_arn: str, n_shot_prompts: int = 0, temperature: float = 0.7, n_shot_prompt_guide_obj: Dict[str, List[Dict[str, str]]] = None):
        """
        Initializes the LlamaInferencer with the given parameters.
        Args:
            model_id (str): The ID of the model to be used for inference
            region (str): The AWS region where the model is hosted
            role_arn (str): The ARN of the IAM role to be used for SageMaker
            n_shot_prompts (int): The number of examples to be used for few-shot prompting
            temperature (float): The temperature parameter for the model's generation
            n_shot_prompt_guide_obj (Dict[str, List[Dict[str, str]]]): A dictionary containing the prompt guide for few-shot prompting
        """
        super().__init__(model_id, region, role_arn, n_shot_prompts, temperature, n_shot_prompt_guide_obj)
        
    def _prepare_conversation(self, message: str, role: str):
        """
        Prepares the conversation format for the model.
        Args:
            message (str): The message to be sent to the model
            role (str): The role of the message sender (e.g., "user", "assistant")
        Returns:
            Dict[str, str]: A dictionary containing the role and content of the message
        """
        # Format message and role into a conversation
        if not message or not role:
            logger.error(f"Error in parsing message or role")
        conversation = {
                "role": role, 
                "content": message
            }
        return conversation
    
    def generate_prompt(self, user_query: str, context: List[Dict]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates the prompt for the model based on the user query and context.
        Args:
            user_query (str): The user's query to be sent to the model
            context (List[Dict]): The context to be used for generating the prompt
        Returns:
            Tuple[str, List[Dict[str, Any]]]: A tuple containing the system prompt and the formatted messages
        """
        # Input validation
        if self.n_shot_prompts < 0:
            raise ValueError("n_shot_prompt must be non-negative")
        
        default_prompt = "You are a helpful assistant. Use the provided context to answer questions accurately. If you cannot find the answer in the context, say so"
        # Get system prompt
        system_prompt = default_prompt if not self.n_shot_prompt_guide_obj or not self.n_shot_prompt_guide_obj.get("system_prompt") else self.n_shot_prompt_guide_obj.get("system_prompt")
        
        context_text = ""
        if context:
            context_text = self.format_context(user_query, context)
        
        base_prompt = self.n_shot_prompt_guide_obj.get("user_prompt", "") if self.n_shot_prompt_guide_obj else ""
        
        if self.n_shot_prompts == 0:
            logger.info("into zero shot prompt")
    
            messages = []
            messages.append(self._prepare_conversation(role="user", message=base_prompt))
            if context_text:
                messages.append(self._prepare_conversation(role="user", message=context_text))
            messages.append(self._prepare_conversation(role="user", message=user_query))

            return system_prompt, messages

        # Get examples if nshot is not zero
        examples = self.n_shot_prompt_guide_obj['examples']
        
        # Format examples
        selected_examples = (random.sample(examples, n_shot_prompt) 
                        if len(examples) > n_shot_prompt 
                        else examples)
        
        logger.info(f"into {n_shot_prompt} shot prompt  with examples {len(selected_examples)}")
        
        messages = []
        messages.append(self._prepare_conversation(role="user", message=base_prompt))
        for example in selected_examples:
            if 'example' in example:
                messages.append(self._prepare_conversation(role="user", message=example['example']))
            elif 'question' in example and 'answer' in example:
                messages.append(self._prepare_conversation(role="user", message=example['question']))
                messages.append(self._prepare_conversation(role="assistant", message=example['answer']))
        
        if context_text:
            messages.append(self._prepare_conversation(role="user", message=context_text))
            
        messages.append(self._prepare_conversation(role="user", message=user_query))

        return system_prompt, messages
        
    def construct_payload(self, system_prompt: str, prompt: str) -> dict:
        """
        Constructs llama 4 payload dictionary for model inference with the given prompts and default parameters.
        
        Args:
            system_prompt (str): The system-level prompt that guides the model's behavior
            prompt (str): The actual prompt/query to be sent to the model
        Returns:
            dict: A dictionary containing the system prompt, user messages, and default parameters for the model
        """
        # Define default parameters for the model's generation
        default_params = {
            "max_new_tokens": 256,
            "temperature": self.temperature,
            "top_p": 0.9,
            "do_sample": True
        }
        
        # Prepare payload for model inference
        payload = {
            "system": system_prompt,
            "messages": prompt,
            "parameters": default_params
            }
        
        return payload
    
    def _extract_response(self, response: dict) -> str:
        """
        Parses the response from the model and extracts the generated text.

        Args:
            response (dict): The raw response from the model
        Returns:
            str: The generated text from the model
        """
        if "choices" in response and isinstance(response["choices"], list):
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unexpected Llama-4 response format: {response}")
        