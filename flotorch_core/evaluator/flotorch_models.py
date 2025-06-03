from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI, AsyncOpenAI
from typing import Tuple


class FloTorchLLMWrapper(DeepEvalBaseLLM):
    def __init__(self, inferencer):
        self.inferencer = inferencer

    def get_model_name(self) -> str:
        return self.inferencer.model_id

    def generate(self, prompt: str) -> Tuple[str, float]:
        client = self.load_model(async_mode=False)
        completion = client.chat.completions.create(
            model=self.inferencer.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        output = completion.choices[0].message.content
        return output

    async def a_generate(self, prompt: str) -> Tuple[str, float]:
        client = self.load_model(async_mode=True)
        completion = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        output = completion.choices[0].message.content
        return output

    def load_model(self, async_mode: bool = False):
        if async_mode:
            return AsyncOpenAI(base_url=self.inferencer.base_url, api_key=self.inferencer.api_key)
        else:
            return OpenAI(base_url=self.inferencer.base_url, api_key=self.inferencer.api_key)
