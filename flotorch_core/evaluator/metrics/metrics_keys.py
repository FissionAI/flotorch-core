from enum import Enum

class MetricKey(str, Enum):
    CONTEXT_PRECISION = "context_precision"
    ASPECT_CRITIC = "aspect_critic"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RECALL="contextual_recall"
    CONTEXT_RELEVANCY="contextual_relevancy"
    HALLUCINATION="hallucination"