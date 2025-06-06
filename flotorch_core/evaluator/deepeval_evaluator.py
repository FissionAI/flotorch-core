from typing import List, Dict, Any, Optional,Type,Union
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from flotorch_core.evaluator.base_evaluator import BaseEvaluator
from flotorch_core.evaluator.evaluation_item import EvaluationItem
from flotorch_core.evaluator.metrics.deepeval_metrics.deepeval_metrics import DeepEvalEvaluationMetrics
from deepeval.models.base_model import DeepEvalBaseLLM
from flotorch_core.inferencer.inferencer import BaseInferencer
from pydantic import BaseModel
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey
from deepeval.models.llms.utils import trim_and_load_json
from flotorch_core.logger.global_logger import get_logger
from itertools import chain

logger = get_logger()


class DeepEvalEvaluator(BaseEvaluator):
    """
    Evaluator that uses DeepEval metrics to evaluate LLM outputs with optional custom metrics
    and support for asynchronous evaluation.

    Initializes with an LLM inferencer and allows configuration of custom metrics, asynchronous execution,
    concurrency limits, and optional metric-specific arguments.
    Args:
        evaluator_llm : The LLM inferencer used for evaluation.
        metric_args :Optional dictionary specifying per-metric configuration arguments.
        Example:
            metric_args = {
                "contextual_recall": {
                    "threshold": 0.6
                },
                "hallucination": {
                    "threshold": 0.4
                }
            }
    """

    def __init__(
        self,
        evaluator_llm: BaseInferencer,
        custom_metrics: Optional[List[Any]] = None,
        async_run: bool = False,
        max_concurrent: int = 1, 
        metric_args: Optional[
            Dict[Union[str, MetricKey], Dict[str, Union[str, float, int]]]
        ] = None

    ):
        class FloTorchLLMWrapper(DeepEvalBaseLLM):
            def __init__(self, inference_llm: BaseInferencer, *args, **kwargs):
                self.inference_llm = inference_llm
                super().__init__(*args, **kwargs)

            def get_model_name(self) -> str:
                """
                Returns the model ID of the underlying inference LLM.
                """
                return self.inference_llm.model_id

            def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> str: 
                """
                Generates a response for a prompt and validates it against a schema if provided.
                """
                client = self.load_model()
                _, completion = client.generate_text(prompt, None)
                return self.schema_validation(completion, schema)

            async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> str: 
                """
                Asynchronously generates a response for a prompt and validates it against a schema if provided.
                """
                client = self.load_model()
                _, completion = await client.generate_text(prompt, None)
                return self.schema_validation(completion, schema)

            def load_model(self):
                """
                Loads and returns the inference LLM client.
                """
                return self.inference_llm
            
            def schema_validation(self, completion: str,schema: Optional[Type[BaseModel]] = None) -> str:
                """
                Validates LLM output against a schema if provided, else returns raw output.
                """
                try:
                    if schema:
                        json_output = trim_and_load_json(completion)
                        return schema.model_validate(json_output)
                    else:
                        return completion
                except Exception as e:
                    logger.error(f"Schema validation error due to {e}.")
                    return completion

        self.llm = FloTorchLLMWrapper(evaluator_llm)
        self.async_config = AsyncConfig(run_async=async_run, max_concurrent=max_concurrent)
        self.custom_metrics = custom_metrics or []
        self.metric_args = metric_args 

        # Initialize DeepEval metrics from the registry
        DeepEvalEvaluationMetrics.initialize_metrics(llm=self.llm, metric_args=self.metric_args)

    def _build_test_cases(self, data: List[EvaluationItem]) -> List[LLMTestCase]:
        """
        Converts evaluation data into LLM test cases for DeepEval evaluation.
        """
        return [
            LLMTestCase(
                input=item.question,
                actual_output=item.generated_answer,
                expected_output=item.expected_answer,
                retrieval_context=item.context or [],
                context=item.context or []
            )
            for item in data
        ]

    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        test_cases = self._build_test_cases(data)
        #example to fetch metrics, use like this
        if metrics is None:
            metrics = DeepEvalEvaluationMetrics.available_metrics()

        selected_metrics = [
            DeepEvalEvaluationMetrics.get_metric(m)
            for m in metrics
        ]
        
        eval_results = evaluate(
            test_cases=test_cases,
            async_config=self.async_config,
            metrics=selected_metrics + self.custom_metrics
        )
        return eval_results.model_dump()

