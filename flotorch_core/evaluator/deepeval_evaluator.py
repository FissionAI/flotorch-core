from typing import List, Dict, Any, Optional, Union,Tuple
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from flotorch_core.evaluator.base_evaluator import BaseEvaluator
from flotorch_core.evaluator.evaluation_item import EvaluationItem
from flotorch_core.evaluator.flotorch_models import FloTorchLLMWrapper
from flotorch_core.evaluator.metrics.deepeval_metrics.deepeval_metrics import DeepEvalEvaluationMetrics

class DeepEvalEvaluator(BaseEvaluator):
    def __init__(
        self,
        inferencer,
        custom_metrics: Optional[List[Any]] = None,
        async_run: bool = False,
        max_concurrent: int = 1,
        metric_args: Optional[Dict[str, Dict[str, Any]]] = None,  # Accept user args

    ):
        self.llm = FloTorchLLMWrapper( 
            inferencer = inferencer           
        )
        
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
        selected_metrics = DeepEvalEvaluationMetrics.get_all_metrics(metrics)

        eval_results = evaluate(
            test_cases=test_cases,
            async_config=self.async_config,
            metrics=selected_metrics + self.custom_metrics
        )
        return eval_results.model_dump()

