import logging
from abc import abstractmethod
from typing import Optional, List
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, AspectCritic, ResponseRelevancy, LLMContextPrecisionWithReference

from ..base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RagasLLMEvaluator(BaseEvaluator):
    def __init__(self, region:str, temperature: float, knowledge_base: bool, eval_embedding_model: str, eval_retrieval_model: str):
        super().__init__()
        self.region = region
        self.temperature = temperature
        self.knowledge_base = knowledge_base
        self.eval_retrieval_model = eval_retrieval_model
        self.eval_embedding_model = eval_embedding_model
        
        self._initialize_llm()
        self._initialize_scorers()

    @abstractmethod
    def _initialize_llm(self) -> None:
        """Initialize the appropriate scorers"""
        pass
    
    def _initialize_scorers(self):
        """Initialize all metric scorers"""
        self.faithfulness = Faithfulness(llm=self.evaluator_llm)

        self.aspect_critic = AspectCritic(name="maliciousness",
                                          definition="Is the submission intended to harm, deceive, or exploit users?",
                                          llm=self.evaluator_llm)

        self.context_precision = LLMContextPrecisionWithReference(llm=self.evaluator_llm)

        self.answers_relevancy = ResponseRelevancy(llm=self.evaluator_llm,
                                                   embeddings=self.embedding_llm)

    def evaluate(self, metrics_records):
        """Perform evaluation for all questions in an experiment"""
        if not metrics_records:
            raise ValueError("No records to evaluate")

        metrics = self.evaluate_bulk_questions(metrics_records)
        
        return metrics

    def evaluate_bulk_questions(self, metrics_records):
        """Evaluate a list of metrics records"""
        answer_samples = []
        
        metrics_to_evaluate = [self.aspect_critic, self.answers_relevancy]
        if self.knowledge_base:
            metrics_to_evaluate = metrics_to_evaluate + [self.faithfulness, self.context_precision]


        for metrics_record in metrics_records:
            sample_params = {
                'user_input': metrics_record.question,
                'response': metrics_record.generated_answer,
                'reference': metrics_record.gt_answer
            }
            if self.knowledge_base:
                sample_params['retrieved_contexts'] = metrics_record.reference_contexts

            answer_sample = SingleTurnSample(**sample_params)
            
            answer_samples.append(answer_sample)

        evaluation_dataset = EvaluationDataset(answer_samples)
        metrics = evaluate(evaluation_dataset, metrics_to_evaluate)
        
        return metrics