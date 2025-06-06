from typing import List, Dict, Any, Optional,Type
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from flotorch_core.evaluator.base_evaluator import BaseEvaluator
from flotorch_core.evaluator.evaluation_item import EvaluationItem
from flotorch_core.evaluator.metrics.deepeval_metrics.deepeval_metrics import DeepEvalEvaluationMetrics
from deepeval.models.base_model import DeepEvalBaseLLM
from flotorch_core.inferencer.inferencer import BaseInferencer
from pydantic import BaseModel
from deepeval.models.llms.utils import trim_and_load_json
from flotorch_core.logger.global_logger import get_logger

logger = get_logger()


class DeepEvalEvaluator(BaseEvaluator):
    def __init__(
        self,
        evaluator_llm: BaseInferencer,
        custom_metrics: Optional[List[Any]] = None,
        async_run: bool = False,
        max_concurrent: int = 1,
        metric_args: Optional[Dict[str, Dict[str, Any]]] = None,  # Accept user args

    ):
        class FloTorchLLMWrapper(DeepEvalBaseLLM):
            def __init__(self, inference_llm: BaseInferencer, *args, **kwargs):
                self.inference_llm = inference_llm
                super().__init__(*args, **kwargs)

            def get_model_name(self) -> str:
                return self.inference_llm.model_id

            def generate(self, prompt: str, schema:Optional[Type[BaseModel]] = None):
                client = self.load_model()
                _, completion = client.generate_text(prompt, None)
                return self.schema_validation(completion, schema)

            async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
                client = self.load_model()
                _, completion = await client.generate_text(prompt, None)
                return self.schema_validation(completion, schema)

            def load_model(self):
                return self.inference_llm
            
            def schema_validation(self, completion: str,schema: Optional[Type[BaseModel]] = None):
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
        self.metric_args = metric_args or {}

        # Initialize DeepEval metrics from the registry
        DeepEvalEvaluationMetrics.initialize_metrics(llm=self.llm, metric_args=self.metric_args)

    def _build_test_cases(self, data: List[EvaluationItem]) -> List[LLMTestCase]:
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
        selected_metrics = DeepEvalEvaluationMetrics.available_metrics(metrics)

        eval_results = evaluate(
            test_cases=test_cases,
            async_config=self.async_config,
            metrics=selected_metrics + self.custom_metrics
        )
        return eval_results.model_dump()

