from typing import List, Dict, Any, Optional, Union
from flotorch_core.evaluator.deepeval_evaluator import DeepEvalEvaluator
from flotorch_core.evaluator.ragas_evaluator import RagasEvaluator
from flotorch_core.inferencer.inferencer import BaseInferencer
from flotorch_core.embedding.embedding import BaseEmbedding
from flotorch_core.evaluator.base_evaluator import BaseEvaluator
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey

class EvaluatorProviderFactory:
    """
    Factory to create evaluator based on the service name.
    """
    @staticmethod
    def create_evaluation_provider(eval_service:str, 
                                   inferencer:BaseInferencer, 
                                   embedding: BaseEmbedding = None, 
                                   metric_args: Optional[Dict[Union[str, MetricKey], Dict[str, Union[str, float, int]]]] = None, 
                                   custom_metrics: Optional[List[Any]] = None, 
                                   async_run: bool = False, 
                                   max_concurrent: int = 1) -> BaseEvaluator:
        if eval_service == 'ragas':
            return RagasEvaluator(evaluator_llm = inferencer,
                                  embedding_llm = embedding,
                                  metric_args = metric_args)
        elif eval_service == 'deepeval':
            return DeepEvalEvaluator(evaluator_llm = inferencer,
                                     custom_metrics = custom_metrics,
                                     async_run = async_run,
                                     max_concurrent = max_concurrent,
                                     metric_args = metric_args
                                     )
        else:
            raise ValueError(f"Unsupported service scheme: {eval_service}")
