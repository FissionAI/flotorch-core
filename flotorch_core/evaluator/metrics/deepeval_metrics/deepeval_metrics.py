from typing import Optional, Dict, Union, Mapping
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
)
# from flotorch_core.evaluator.custom_metrics import CustomMetric
from flotorch_core.evaluator.metrics.deepeval_metrics_keys import MetricKey

class DeepEvalEvaluationMetrics:
    _registry = {
        MetricKey.FAITHFULNESS: {
            "class": FaithfulnessMetric,
            "default_args": {"threshold": 0.7, "truths_extraction_limit": 50}
        },
        MetricKey.CONTEXT_RELAVENCY: {
            "class": ContextualRelevancyMetric,
            "default_args": {"threshold": 0.7}
        },
        MetricKey.CONTEXT_PRECISION: {
            "class": ContextualPrecisionMetric,
            "default_args": {"threshold": 0.7}
        },
        MetricKey.CONTEXT_RECALL: {
            "class": ContextualRecallMetric,
            "default_args": {"threshold": 0.7}
        },
        MetricKey.ANSWER_RELEVANCE: {
            "class": AnswerRelevancyMetric,
            "default_args": {"threshold": 0.7}
        },
        MetricKey.HALLUCINATION: {
            "class": HallucinationMetric,
            "default_args": {"threshold": 0.5}
        },
        # "custom_metric": {
        #     "class": CustomMetric,
        #     "default_args": {"threshold": 0.5}
        # },
    }

    _initialized_metrics: Dict[str, object] = {}

    @classmethod
    def initialize_metrics(
        cls,
        llm,
        metric_args: Optional[Mapping[str, Dict[str, Union[float, int]]]] = None,
    ):
        cls._initialized_metrics = {}
        metric_args = metric_args or {}

        for name, config in cls._registry.items():
            args = config["default_args"].copy()
            args.update(metric_args.get(name, {}))  # override defaults
            cls._initialized_metrics[name] = config["class"](model=llm, **args)

    @classmethod
    def get_metric(cls, name: str):
        if name not in cls._initialized_metrics:
            raise ValueError(f"Metric '{name}' has not been initialized.")
        return cls._initialized_metrics[name]

    @classmethod
    def get_all_metrics(cls, selected: Optional[list[str]] = None):
        selected = selected or list(cls._initialized_metrics.keys())
        return [cls.get_metric(name) for name in selected]
