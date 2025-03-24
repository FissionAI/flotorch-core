import random
from openai import OpenAI
from typing import List, Dict, Tuple

from flotorch_core.inferencer.inferencer import BaseInferencer
import time


class GatewayInferencer(BaseInferencer):
    def __init__(self, model_id: str, api_key: str, base_url: str = None, n_shot_prompts: int = 0, n_shot_prompt_guide_obj: Dict[str, List[Dict[str, str]]] = None):
        super().__init__(model_id, None, n_shot_prompts, None, n_shot_prompt_guide_obj)
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_prompt(self, user_query: str, context: List[Dict]) -> List[Dict[str, str]]:
        messages = []

        default_prompt = "You are a helpful assistant. Use the provided context to answer questions accurately. If you cannot find the answer in the context, say so"
        
        system_prompt = (
            self.n_shot_prompt_guide_obj.get("system_prompt", default_prompt)
            if self.n_shot_prompt_guide_obj
            else default_prompt
        )
        messages.append({"role": "system", "content": system_prompt})

        if context:
            context_text = self.format_context(context)
            if context_text:
                messages.append({"role": "user", "content": context_text})

        base_prompt = self.n_shot_prompt_guide_obj.get("user_prompt", "") if self.n_shot_prompt_guide_obj else ""
        if base_prompt:
            messages.append({"role": "user", "content": base_prompt})

        if self.n_shot_prompt_guide_obj:
            examples = self.n_shot_prompt_guide_obj.get("examples", [])
            selected_examples = (
                random.sample(examples, self.n_shot_prompts)
                if len(examples) > self.n_shot_prompts
                else examples
            )
            for example in selected_examples:
                if "example" in example:
                    messages.append({"role": "user", "content": example["example"]})
                elif "question" in example and "answer" in example:
                    messages.append({"role": "user", "content": example["question"]})
                    messages.append({"role": "assistant", "content": example["answer"]})

        messages.append({"role": "user", "content": user_query})

        return messages

    def generate_text(self, user_query: str, context: List[Dict]) -> Tuple[Dict, str]:
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
        raise NotImplementedError("Subclasses must implement format_context")
    
    def _extract_metadata(self, response):
        return {
            "inputTokens": str(response.usage.prompt_tokens),
            "outputTokens": str(response.usage.completion_tokens),
            "totalTokens": str(response.usage.total_tokens),
            "latencyMs": "0"
        }