from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI, AsyncOpenAI
from typing import Tuple


class FloTorchLLMWrapper(DeepEvalBaseLLM):
    def __init__(self, inferencer,base_url,_openai_api_key,model_name):
        self.inferencer = inferencer
        self._openai_api_key = _openai_api_key
        self.base_url =base_url
        self.model_name = model_name
        self.temperature = 0

    def get_model_name(self) -> str:
        return self.inferencer.model_id

    def generate(self, prompt: str) -> Tuple[str, float]:
        client = self.load_model(async_mode=False)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        output = completion.choices[0].message.content
        return output

    async def a_generate(self, prompt: str) -> Tuple[str, float]:
        client = self.load_model(async_mode=True)
        completion = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        output = completion.choices[0].message.content
        return output

    def load_model(self, async_mode: bool = False):
        if async_mode:
            return AsyncOpenAI(base_url=self.base_url, api_key=self._openai_api_key)
        else:
            return OpenAI(base_url=self.base_url, api_key=self._openai_api_key)
