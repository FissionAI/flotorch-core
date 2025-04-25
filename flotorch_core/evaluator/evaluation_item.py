from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvaluationItem:
    """
    Represents one unit of evaluation — a question and its corresponding model output and metadata.
    """
    question: str
    generated_answer: str
    expected_answer: str
    context: Optional[List[str]] = None
