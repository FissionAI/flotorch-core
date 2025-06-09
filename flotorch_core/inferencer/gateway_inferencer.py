import random
from openai import OpenAI
from typing import List, Dict, Tuple
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.inferencer.inferencer import BaseInferencer, DEFAULT_SYSTEM_PROMPT
import time

logger = get_logger()

class GatewayInferencer(BaseInferencer):
    def __init__(self, model_id: str, api_key: str, base_url: str = None, n_shot_prompts: int = 0, n_shot_prompt_guide_obj: Dict[str, List[Dict[str, str]]] = None):
        super().__init__(model_id, None, n_shot_prompts, None, n_shot_prompt_guide_obj)
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_prompt(self, user_query: str, use_system: bool, context: List[Dict]) -> List[Dict[str, str]]:
        messages = []
        system_prompt = None
        # System prompt
        if use_system:
            system_prompt = self.n_shot_prompt_guide_obj.get("system_prompt", "") if self.n_shot_prompt_guide_obj and self.n_shot_prompt_guide_obj.get("system_prompt") else DEFAULT_SYSTEM_PROMPT
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

    def generate_text(self, user_query: str, context: List[Dict], use_system: bool = True) -> Tuple[Dict, str]:
        messages  = self.generate_prompt(user_query, use_system, context)
        
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
        """
        return "\n".join([f"Context {i+1}:\n{item['text']}" for i, item in enumerate(context)])
    
    def _extract_metadata(self, response):
        return {
            "inputTokens": str(response.usage.prompt_tokens),
            "outputTokens": str(response.usage.completion_tokens),
            "totalTokens": str(response.usage.total_tokens),
            "latencyMs": "0"
        }