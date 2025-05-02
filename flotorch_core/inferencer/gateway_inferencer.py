import random
from openai import OpenAI
from typing import List, Dict, Tuple

from flotorch_core.inferencer.inferencer import BaseInferencer
import time


class GatewayInferencer(BaseInferencer):
    """
    GatewayInferencer is a class that interacts with the OpenAI API to generate text completions.
    It uses the OpenAI Python client to send requests and receive responses.
    """
    def __init__(self, model_id: str, api_key: str, base_url: str = None, n_shot_prompts: int = 0, n_shot_prompt_guide_obj: Dict[str, List[Dict[str, str]]] = None):
        """
        Initialize the GatewayInferencer with the model ID, API key, and base URL.
        Args:
            model_id (str): The ID of the model to use for text generation.
            api_key (str): The API key for authentication with the OpenAI API.
            base_url (str): The base URL for the OpenAI API. Defaults to None.
            n_shot_prompts (int): The number of examples to include in the prompt.
            n_shot_prompt_guide_obj (Dict[str, List[Dict[str, str]]]): A dictionary containing the prompt guide object.
        """
        super().__init__(model_id, None, n_shot_prompts, None, n_shot_prompt_guide_obj)
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_prompt(self, user_query: str, context: List[Dict]) -> List[Dict[str, str]]:
        """
        Generate a prompt for the OpenAI API based on the user query and context.
        Args:
            user_query (str): The user's query to be included in the prompt.
            context (List[Dict]): A list of context items to be included in the prompt.
        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the messages to be sent to the OpenAI API.
        """
        messages = []
        
        # System prompt
        default_prompt = "You are a helpful assistant. Use the provided context to answer questions accurately. If you cannot find the answer in the context, say so"
        system_prompt = (
            self.n_shot_prompt_guide_obj.get("system_prompt", default_prompt)
            if self.n_shot_prompt_guide_obj
            else default_prompt
        )
        messages.append({"role": "system", "content": system_prompt})

        # Nshot examples
        if self.n_shot_prompt_guide_obj:
            examples = self.n_shot_prompt_guide_obj.get("examples", [])
            selected_examples = (
                random.sample(examples, self.n_shot_prompts)
                if len(examples) > self.n_shot_prompts
                else examples
            )
            for example in selected_examples:
                if "example" in example:
                    messages.append({"role": "assistant", "content": example["example"]})
                elif "question" in example and "answer" in example:
                    messages.append({"role": "user", "content": example["question"]})
                    messages.append({"role": "assistant", "content": example["answer"]})
             
        # Context
        if context:
            context_text = self.format_context(context)
            if context_text:
                messages.append({"role": "user", "content": context_text})

        # User query and base prompt
        base_prompt = self.n_shot_prompt_guide_obj.get("user_prompt", "") if self.n_shot_prompt_guide_obj else ""
        # Combine base prompt with user query if base prompt is provided else use user query
        query = base_prompt + "\n" + user_query if base_prompt else user_query
        messages.append({"role": "user", "content": query})
        
        return messages

    def generate_text(self, user_query: str, context: List[Dict]) -> Tuple[Dict, str]:
        """
        Generate text using the OpenAI API based on the user query and context.
        Args:
            user_query (str): The user's query to be included in the prompt.
            context (List[Dict]): A list of context items to be included in the prompt.
        Returns:
            Tuple[Dict, str]: A tuple containing metadata and the generated text.
        """
        messages  = self.generate_prompt(user_query, context)
        
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages
        )


        end_time = time.time()

        metadata = self._extract_metadata(response)
        metadata["latencyMs"] = str(int((end_time - start_time) * 1000))
        
        return metadata, response.choices[0].message.content


    def format_context(self, context: List[Dict[str, str]]) -> str:
        """
        Format context into a string to be included in the prompt.
        Args:
            context (List[Dict[str, str]]): A list of context items to be formatted.
        Returns:
            str: A formatted string representing the context.
        """
        return "\n".join([f"Context {i+1}:\n{item['text']}" for i, item in enumerate(context)])
    
    def _extract_metadata(self, response):
        """
        Extract metadata from the OpenAI API response.
        Args:
            response: The response object from the OpenAI API.
        Returns:
            Dict: A dictionary containing metadata such as input tokens, output tokens, and total tokens.
        """
        return {
            "inputTokens": str(response.usage.prompt_tokens),
            "outputTokens": str(response.usage.completion_tokens),
            "totalTokens": str(response.usage.total_tokens),
            "latencyMs": "0"
        }