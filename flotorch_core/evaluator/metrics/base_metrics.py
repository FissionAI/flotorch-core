from abc import ABC, abstractmethod
from typing import List

class BaseEvaluationMetric(ABC):
    """
    Abstract base class to define metric registries.
    """
    @classmethod
    @abstractmethod
    def available_metrics(cls) -> List[str]:
        """
        Returns a list of available metrics.
        """
        pass

    @classmethod
    @abstractmethod
    def get_metric(cls, key: str):
        """
        Returns the metric associated with the key.
        """
        pass

    @classmethod
    @abstractmethod
    def initialize_metrics(cls, llm, embeddings):
        """
        Initializes the metrics with the provided LLM and embeddings.
        """
        pass