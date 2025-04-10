from typing import Any, Dict, List, Optional

from flotorch_core.evaluator.base_evaluator import BaseEvaluator
from flotorch_core.evaluator.evaluation_item import EvaluationItem
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey
from flotorch_core.evaluator.metrics.ragas_metrics.ragas_metrics import RagasEvaluationMetrics


class RagasEvaluator(BaseEvaluator):
    """
    Evaluator that uses RAGAS metrics to score RAG-based QA performance.
    """
    def __init__(self, evaluator_llm, embedding_llm):
        self.evaluator_llm = evaluator_llm
        self.embedding_llm = embedding_llm

        # TODO: wrap with internal class which extends base class which ragas uses for llm, then pass those in the below metrics instead
        
        RagasEvaluationMetrics.initialize_metrics(
            llm=self.evaluator_llm,
            embeddings=self.embedding_llm
        )

    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Optional[List[MetricKey]] = None
    ) -> Dict[str, Any]:
        # example to fetch metrics, use like this
        faithfulness_metric = RagasEvaluationMetrics.get_metric(MetricKey.FAITHFULNESS)
        pass