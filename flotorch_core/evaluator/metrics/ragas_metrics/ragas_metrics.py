from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    AspectCritic,
)
from flotorch_core.evaluator.metrics.base_metrics import BaseEvaluationMetric
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey


class RagasEvaluationMetrics(BaseEvaluationMetric):
    """
    Registry of RAGAS metric classes and their initialized instances.
    """
    _registry = {
        MetricKey.CONTEXT_PRECISION: {"class": LLMContextPrecisionWithReference, "requires": ["llm"]},
        MetricKey.ASPECT_CRITIC: {"class": AspectCritic, "requires": ["llm", "name", "definition"]},
        MetricKey.FAITHFULNESS: {"class": Faithfulness, "requires": ["llm"]},
        MetricKey.ANSWER_RELEVANCE: {"class": ResponseRelevancy, "requires": ["llm", "embeddings"]},
    }

    _initialized_metrics = {}

    @classmethod
    def available_metrics(cls):
        return list(cls._registry.keys())

    @classmethod
    def initialize_metrics(cls, llm, embeddings, metric_args=None):
        """
        Initializes all metric objects and stores them internally.
        """
        cls._initialized_metrics = {}
        metric_args = metric_args or {}
        for key, info in cls._registry.items():
            args = {}

            if "llm" in info["requires"]:
                args["llm"] = llm
            if "embeddings" in info["requires"]:
                args["embeddings"] = embeddings
                
            # Check if name and definition are required
            if "name" in info["requires"] or "definition" in info["requires"]:
                metric_categories = metric_args.get(key, {})
                for category_name, category_config in metric_categories.items():
                    if "name" not in category_config or "definition" not in category_config:
                        raise ValueError(f"Metric '{key}' requires 'name' and 'definition' in metric_args.")
                    args["name"] = category_config["name"]
                    args["definition"] = category_config["definition"]

                    cls._initialized_metrics[f"{key}_{category_name}"] = info["class"](**args)
            cls._initialized_metrics[key] = info["class"](**args)

    @classmethod
    def get_metric(cls, key: str):
        """
        Returns the initialized metric object.
        """
        if key not in cls._initialized_metrics:
            raise ValueError(f"Metric '{key}' has not been initialized. Did you forget to call `initialize_metrics()`?")
        return cls._initialized_metrics[key]
