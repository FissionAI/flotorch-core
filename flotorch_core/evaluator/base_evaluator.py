from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from flotorch_core.evaluator.evaluation_item import EvaluationItem
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey

class BaseEvaluator(ABC):
    """
    Abstract base class for evaluation modules.
    """

    @abstractmethod
    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Optional[List[MetricKey]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model output against the expected answers.

        Args:
            data (List[EvaluationItem]): List of evaluation inputs.

        Returns:
            Dict[str, Any]: Dictionary of evaluation results.
        """
        pass
