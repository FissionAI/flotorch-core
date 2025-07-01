from typing import Optional, Dict, Union, Mapping
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
)
from flotorch_core.evaluator.metrics.base_metrics import BaseEvaluationMetric
# from flotorch_core.evaluator.custom_metrics import CustomMetric
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey

class DeepEvalEvaluationMetrics(BaseEvaluationMetric):
    _registry = {
        MetricKey.FAITHFULNESS: {
            "class": FaithfulnessMetric,
            "default_args": {"threshold": 0.7, "truths_extraction_limit": 30}
        },
        MetricKey.CONTEXT_RELEVANCY: {
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
    def available_metrics(cls) -> list[str]:
        return list(cls._initialized_metrics.keys())

    @classmethod
    def initialize_metrics(
        cls,
        llm,
        metric_args: Optional[
            Mapping[Union[str, MetricKey], Dict[str, Union[str, float, int]]]
        ] = None,
    ):
        """
        Initializes metric instances and stores them in an internal dictionary.

        This method iterates over the registered metrics in `cls._registry`, initializes each metric,
        and stores it in `cls._initialized_metrics`.  
        You can optionally override default arguments for each metric using `metric_args`.

        Args:
            llm: A language model wrapper instance (used as `model` argument for each metric class).
            metric_args (Optional): A dictionary providing argument overrides for each metric.
                The structure is:

                {
                    "metric_name": {
                        "arg1": value1,
                        "arg2": value2,
                        ...
                    },
                    ...
                }

                - The outer key corresponds to a metric name (as used in `cls._registry`).
                - The inner dictionary provides argument overrides that will replace defaults.

        Example Usage:
            metric_args={
                "faithfulness": {
                "threshold": 0.8
                }                    
            }
            
        """

        cls._initialized_metrics = {}
        metric_args = metric_args or {} 

        for name, config in cls._registry.items():
            args = config["default_args"].copy()
            args.update(metric_args.get(name, {}))  # override defaults
            cls._initialized_metrics[name] = config["class"](model=llm, **args)

    @classmethod
    def get_metric(cls, name: str) -> Dict[str, object]:
        """
        Retrieves an initialized metric instance by name.

        Args:
            name (str): The name of the metric to retrieve (must match a name in `cls._registry`).

        Returns:
            object: The initialized metric instance.

        Raises:
            ValueError: If the requested metric has not been initialized (i.e. not present in `cls._initialized_metrics`).

        Example Usage:
            faithfulness_metric = get_metric("faithfulness")
        """
        if name not in cls._initialized_metrics:
            raise ValueError(f"Metric '{name}' has not been initialized.")
        return cls._initialized_metrics[name]

    
        