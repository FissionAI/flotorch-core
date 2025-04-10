import logging
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from flotorch_core.evaluation.ragas.ragas_llm_evaluator import RagasLLMEvaluator
from ..eval_factory import register

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@register('ragas', 'bedrock')
class RagasLLMBedrockEvaluator(RagasLLMEvaluator):
    def __init__(self, region: str, temperature: int, eval_embedding_model: str, eval_retrieval_model:str, knowledge_base: bool):
        
        super().__init__(region, temperature, knowledge_base, eval_embedding_model, eval_retrieval_model)


    def _initialize_llm(self):
        try:
            self.evaluator_llm = LangchainLLMWrapper(ChatBedrockConverse(
                    region_name=self.region,
                    base_url=f"https://bedrock-runtime.{self.region}.amazonaws.com",
                    model=self.eval_retrieval_model,
                    temperature=self.temperature,
                    ))

            self.embedding_llm = LangchainEmbeddingsWrapper(BedrockEmbeddings(
                region_name=self.region,
                model_id=self.eval_embedding_model,
            ))
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")