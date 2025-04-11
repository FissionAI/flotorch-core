import asyncio
from typing import Any, Dict, List, Optional

from flotorch_core.chunking.chunking import Chunk
from flotorch_core.embedding.embedding import BaseEmbedding
from flotorch_core.evaluator.base_evaluator import BaseEvaluator
from flotorch_core.evaluator.evaluation_item import EvaluationItem
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey
from flotorch_core.evaluator.metrics.ragas_metrics.ragas_metrics import RagasEvaluationMetrics

from langchain.embeddings.base import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_core.language_models.base import LanguageModelLike
from ragas.evaluation import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset


from langchain_openai import OpenAIEmbeddings

from flotorch_core.inferencer.inferencer import BaseInferencer

class RagasEvaluator(BaseEvaluator):
    """
    Evaluator that uses RAGAS metrics to score RAG-based QA performance.
    """
    def __init__(self, evaluator_llm: BaseInferencer, embedding_llm: BaseEmbedding):
        self.evaluator_llm = evaluator_llm
        self.embedding_llm = embedding_llm

        # TODO: wrap with internal class which extends base class which ragas uses for llm, then pass those in the below metrics instead
        class _EmbeddingWrapper(Embeddings):
            def __init__(self, internal_embedding):
                self.internal_embedding = internal_embedding

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self._embed_text(text) for text in texts]

            def embed_query(self, text: str) -> list[float]:
                return self._embed_text(text)

            def _embed_text(self, text: str) -> list[float]:
                chunk = Chunk(data=text)
                embedding = self.internal_embedding.embed(chunk)
                return embedding.embeddings[0]
            
        class _LLMWrapper(LanguageModelLike):
            def __init__(self, internal_llm: BaseInferencer):
                self.internal_llm = internal_llm

            def invoke(self, prompt: str) -> str:
                """
                This mimics LangChain's ChatOpenAI behavior.
                """
                metadata, response = self.internal_llm.generate_text(user_query=prompt, context=[])
                return response
            
            async def ainvoke(self, prompt: str) -> str:
                """
                Async interface — RAGAS prefers this if available.
                """
                # Run the sync method in an async wrapper
                return await asyncio.to_thread(self.invoke, prompt)
            
        wrapped_embedding = LangchainEmbeddingsWrapper(_EmbeddingWrapper(self.embedding_llm))
        wrapped_evaluator_llm = LangchainLLMWrapper(_LLMWrapper(self.evaluator_llm))
        
        RagasEvaluationMetrics.initialize_metrics(
            llm=wrapped_evaluator_llm,
            embeddings=wrapped_embedding
        )


    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Optional[List[MetricKey]] = None
    ) -> Dict[str, Any]:
        # example to fetch metrics, use like this
        if metrics is None:
            metrics = RagasEvaluationMetrics.available_metrics()

        selected_metrics = [RagasEvaluationMetrics.get_metric(m) for m in metrics]

        answer_samples = []
        for item in data:
            sample_params = {
                "user_input": item.question,
                "response": item.generated_answer,
                "reference": item.expected_answer,
            }
            if item.context:
                sample_params["retrieved_contexts"] = item.context

            answer_samples.append(SingleTurnSample(**sample_params))

        evaluation_dataset = EvaluationDataset(answer_samples)

        result = evaluate(evaluation_dataset, selected_metrics)