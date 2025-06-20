import asyncio
from typing import Any, Dict, List, Optional, Union

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
from langchain_core.outputs.generation import Generation
from ragas.evaluation import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from langchain_core.outputs import LLMResult
from itertools import chain

from flotorch_core.inferencer.inferencer import BaseInferencer

class RagasEvaluator(BaseEvaluator):
    """
    Evaluator that uses RAGAS metrics to score RAG-based QA performance.
    Initializes the RagasEvaluator with the given LLM and embedding wrappers.

    Args:
        evaluator_llm: The LLM to be used by RAGAS metrics (wrapped in LangchainLLMWrapper).
        embedding_llm: The embedding model to be used by RAGAS metrics (wrapped in LangchainEmbeddingsWrapper).
        metric_args: Optional configuration for metrics requiring per-instance arguments.

            Example:
            {
                MetricKey.ASPECT_CRITIC: {
                    "maliciousness": {
                        "name": "maliciousness",
                        "definition": "Is the response harmful?"
                    },
                    "bias": {
                        "name": "bias",
                        "definition": "Is the response biased or discriminatory?"
                    }
                }
            }
    """
    def __init__(
        self,
        evaluator_llm: BaseInferencer,
        embedding_llm: BaseEmbedding,
        metric_args: Optional[
            Dict[Union[str, MetricKey], Dict[str, Dict[str, str]]]
        ] = None
    ):
        self.evaluator_llm = evaluator_llm
        self.embedding_llm = embedding_llm
        self.metric_args = metric_args

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
                return embedding.embeddings
            
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
        
            def generate_prompt(self, prompts: List[str], **kwargs: Any,):
                """
                Sync implementation for prompt generation
                """
                responses = []
                for prompt in prompts:
                    metadata, response = self.internal_llm.generate_text(
                        user_query=prompt.text, 
                        context=[]
                    )
                    responses.append(response)
                
                return LLMResult(generations=[[Generation(text=resp)] for resp in responses])

            async def agenerate_prompt(self, prompts: List[str], **kwargs: Any,) -> LLMResult:
                loop = asyncio.get_event_loop()
                futures = []
                
                for prompt in prompts:
                    futures.append(
                        loop.run_in_executor(
                            None, 
                            self.internal_llm.generate_text,
                            prompt.text,
                            []
                        )
                    )
                    
                results = await asyncio.gather(*futures)
                responses = [resp[1] for resp in results]
                
                return LLMResult(generations=[[Generation(text=resp)] for resp in responses])
        
        wrapped_embedding = LangchainEmbeddingsWrapper(_EmbeddingWrapper(self.embedding_llm))
        wrapped_evaluator_llm = LangchainLLMWrapper(_LLMWrapper(self.evaluator_llm))
        
        RagasEvaluationMetrics.initialize_metrics(
            llm=wrapped_evaluator_llm,
            embeddings=wrapped_embedding,
            metric_args=self.metric_args
        )


    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Optional[List[MetricKey]] = None
    ) -> Dict[str, Any]:
        # example to fetch metrics, use like this
        if metrics is None:
            metrics = RagasEvaluationMetrics.available_metrics()

        selected_metrics = list(chain.from_iterable(
            RagasEvaluationMetrics.get_metric(m).values() for m in metrics
        ))

        answer_samples = []
        for item in data:
            sample_params = {
                "user_input": item.question,
                "response": item.generated_answer,
                "reference": item.expected_answer,
                "retrieved_contexts": item.context
            }
            answer_samples.append(SingleTurnSample(**sample_params))

        evaluation_dataset = EvaluationDataset(answer_samples)

        result = evaluate(evaluation_dataset, selected_metrics)
        
        return result
