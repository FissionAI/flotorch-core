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
        Base method to return a list of available metrics.
        Returns a list of available metrics.
        """
        pass

    @classmethod
    @abstractmethod
    def get_metric(cls, key: str):
        """
        Base method to return a metric associated with the key.
        Args:
            key (str): The key associated with the metric.
        Returns the metric associated with the key.
        """
        pass

    @classmethod
    @abstractmethod
    def initialize_metrics(cls, llm, embeddings):
        """
        Initializes the metrics with the provided LLM and embeddings.
        Args:
            llm: The LLM to be used for evaluation.
            embeddings: The embeddings to be used for evaluation.
        """
        pass