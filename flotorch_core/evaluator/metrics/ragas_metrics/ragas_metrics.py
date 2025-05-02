from typing import Union, Optional, Mapping, Dict

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
        MetricKey.CONTEXT_PRECISION: {
            "class": LLMContextPrecisionWithReference,
            "requires": ["llm"]
        },
        MetricKey.ASPECT_CRITIC: {
            "class": AspectCritic,
            "requires": ["llm"],
            "metric_args": ["name", "definition"]
        },
        MetricKey.FAITHFULNESS: {
            "class": Faithfulness,
            "requires": ["llm"]
        },
        MetricKey.ANSWER_RELEVANCE: {
            "class": ResponseRelevancy,
            "requires": ["llm", "embeddings"]
        },
    }

    """
    _initialized_metrics: A dictionary like:
        {
            "aspect_critic": {
                "maliciousness": <AspectCritic(...)>,
                "bias": <AspectCritic(...)>
            },
            "faithfulness": {
                "default": <Faithfulness(...)>
            },
            ...
        }
    """
    _initialized_metrics: Dict[str, Dict[str, object]] = {}

    @classmethod
    def available_metrics(cls) -> list[str]:
        """
        Returns a list of available metrics.
        Returns:
            list[str]: A list of metric keys.
        """
        return list(cls._initialized_metrics.keys())

    @classmethod
    def initialize_metrics(
        cls,
        llm,
        embeddings,
        metric_args: Optional[
            Mapping[Union[str, MetricKey], Dict[str, Dict[str, str]]]
        ] = None,
    ):
        """
        Initializes metrics based on the registry and stores them in a dict:
        Args:
            llm: A language model wrapper (e.g., LangchainLLMWrapper)
            embeddings: An embedding wrapper (e.g., LangchainEmbeddingsWrapper)
            metric_args (Optional): A nested dictionary specifying arguments for
                per-instance configuration of metrics that require additional input.
                The structure should be:

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
        cls._initialized_metrics = {}
        metric_args = metric_args or {}

        for key, config in cls._registry.items():
            base_args = {}
            if "llm" in config["requires"]:
                base_args["llm"] = llm
            if "embeddings" in config["requires"]:
                base_args["embeddings"] = embeddings

            metric_class = config["class"]
            key_str = key.value

            if "metric_args" in config:
                arg_map = metric_args.get(key, {})
                cls._initialized_metrics[key_str] = {}

                for identifier, arg_config in arg_map.items():
                    missing = [param for param in config["metric_args"] if param not in arg_config]
                    if missing:
                        raise ValueError(f"Metric '{key}' is missing required args: {', '.join(missing)}")

                    full_args = base_args | {param: arg_config[param] for param in config["metric_args"]}
                    cls._initialized_metrics[key_str][identifier] = metric_class(**full_args)

            else:
                cls._initialized_metrics[key_str] = {
                    "default": metric_class(**base_args)
                }

    @classmethod
    def get_metric(cls, key: Union[str, MetricKey]) -> Dict[str, object]:
        """
        Returns a dictionary of initialized metric instances for the given metric key.

        Args:
            key (Union[str, MetricKey]): The metric key to fetch (e.g., MetricKey.FAITHFULNESS).

        Returns:
            Dict[str, object]: A dictionary where:
                - Each key is an identifier (e.g., "default", "maliciousness", "bias")
                - Each value is an initialized RAGAS metric instance

        Example:
            {
                "aspect_critic": {
                    "maliciousness": <AspectCritic(...)>,
                    "bias": <AspectCritic(...)>
                },
                "faithfulness": {
                    "default": <Faithfulness(...)>
                }
            }

        Raises:
            ValueError: If the metric key has not been initialized.
        """
        if isinstance(key, MetricKey):
            key = key.value

        if key not in cls._initialized_metrics:
            raise ValueError(f"Metric '{key}' has not been initialized. Did you forget to call `initialize_metrics()`?")
        return cls._initialized_metrics[key]
