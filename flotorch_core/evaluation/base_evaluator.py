from abc import ABC,abstractclassmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
        
        
class BaseEvaluator(ABC):
    """Abstract base class for evaluators.
    
    This class defines the interface for evaluator classes that perform experiment evaluation.

    """

    def __init__(self):
        """Initialize evaluator.
        
        Args:
            region (str): AWS region where evaluation is performed
            experiment_id (str): Unique identifier for the experiment
        """
        pass

    @abstractclassmethod
    def _initialize_scorers() -> None:
        """Initialize scoring parameters.
        
        This method should be implemented by subclasses to set up any required scoring components.
        """
        pass
    
    @abstractclassmethod
    def evaluate(experiment_id: str) -> None:
        """Evaluate an experiment.
        
        Args:
            experiment_id (str): Unique identifier for the experiment to evaluate
        """
        pass
    
    @abstractclassmethod
    def evaluate_bulk_questions(metrics_records):
        """Evaluate multiple experiment questions in bulk.
        
        Args:
            metrics_records (List[ExperimentQuestionMetrics]): List of question metrics to evaluate
        """
        pass
    
