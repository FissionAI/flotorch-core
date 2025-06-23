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
from deepeval.evaluate import ErrorConfig
from tenacity import retry, wait_exponential_jitter, retry_if_exception_type, stop_after_attempt
import json
logger = get_logger()


class DeepEvalEvaluator(BaseEvaluator):
    """
    Evaluator that uses DeepEval metrics to evaluate LLM outputs with optional custom metrics
    and support for asynchronous evaluation.

    Initializes with an LLM inferencer and allows configuration of custom metrics, asynchronous execution,
    concurrency limits, and optional metric-specific arguments.
    Args:
        evaluator_llm : The LLM inferencer used for evaluation.
        custom_metrics :A list of additional metric instances to include in evaluation beyond the default DeepEval metrics registry.
        async_run :Whether to run evaluation asynchronously.If True, evaluation can run concurrently up to `max_concurrent` tasks.
        max_concurrent : Maximum number of concurrent asynchronous evaluation tasks to run.    
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
            
           
            @retry(
                wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
                retry=retry_if_exception_type(ValueError),
                stop=stop_after_attempt(3)
            )
            def schema_validation(self, completion: str, schema: Optional[Type[BaseModel]] = None) -> str:
                try:
                    if schema:
                        json_output = self.trim_json(completion)
                        parsed_output = json.loads(json_output)
                        return schema.model_validate(parsed_output)
                    else:
                        return completion
                except ValueError as ve:
                    raise ve  
                except Exception as e:
                    logger.error(f"Schema validation error due to {e}.")
                    return completion

                
            def llm_fix_json_prompt(self, bad_json: str) -> str:
                return f"""The following is a malformed JSON (possibly incomplete or with syntax issues). Fix it so that it becomes **valid JSON**.
                        Instructions:
                        - Do **NOT** include Markdown formatting (no triple backticks, no ```json).
                        - Do **NOT** add or invent any new keys or values.
                        - Only fix unclosed strings, arrays, or braces.
                        - Do **NOT** add commas or fields that were not originally present.
                        - **Remove any trailing commas** at the end of JSON objects or arrays.
                        - **Ensure all property names and string values are enclosed in double quotes**.
                        - Preserve the original structure and values.
                        - If a list like "truths" or "verdicts" or "statements" or "reason" seems incomplete, just close it properly.
                        - Output **only** valid, raw JSON. No explanation, no surrounding text, no markdown.

                        Malformed JSON to fix:
                        {json.dumps(bad_json)}
                        """
            def fix_common_truncation(self,json_str: str) -> str:
                if not json_str.endswith(']') and not json_str.endswith('}'):
                    json_str += '"}]}'  
                return json_str
            
            def trim_json(self, completion: str) -> str:
                client = self.load_model()  # Load the model like in `generate()`

                prompt = self.llm_fix_json_prompt(completion)

                # Assuming client has a method similar to `generate_text(prompt, None)`
                _, fixed_json = client.generate_text(prompt, None,False)

                fixed_json = fixed_json.strip()

                # Optional: Validate the output is valid JSON
                try:
                    json.loads(fixed_json)
                except json.JSONDecodeError as e:
                    logger.warning("Detected JSON truncation. Trying naive fix.")
                    fixed_json = self.fix_common_truncation(fixed_json)
                    try:
                        json.loads(fixed_json)
                    except json.JSONDecodeError as e2:
                        raise ValueError(f"Model returned invalid JSON (even after fix): {e2}\n\nReturned:\n{fixed_json}")
                return fixed_json





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
            metrics=selected_metrics + self.custom_metrics,
            error_config=ErrorConfig(ignore_errors=True)
        )
        return eval_results.model_dump()

