from enum import Enum

class MetricKey(str, Enum):
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
