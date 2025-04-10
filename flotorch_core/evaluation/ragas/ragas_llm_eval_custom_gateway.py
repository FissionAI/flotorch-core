from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from flotorch_core.evaluation.ragas.ragas_llm_evaluator import RagasLLMEvaluator
from ..eval_factory import register

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@register('ragas', 'custom_gateway')
class RagasLLMEvaluatorCustomGateway(RagasLLMEvaluator):
    def __init__(self, region: str, temperature: str, eval_embedding_model: str, eval_retrieval_model: str, knowledge_base: bool, gateway_url: str, gateway_api_key: str):
        
        self.gateway_url = gateway_url
        self.gateway_api_key = gateway_api_key
        super().__init__(region, temperature, knowledge_base, eval_embedding_model, eval_retrieval_model)

    def _initialize_llm(self):
        try:
            self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
                base_url=self.gateway_url,
                api_key=self.gateway_api_key,
                model=self.eval_retrieval_model,
                temperature=self.temperature,
            ))

            self.embedding_llm = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
                model=self.eval_embedding_model,
                base_url=self.gateway_url,
                api_key=self.gateway_api_key,
            ))
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")