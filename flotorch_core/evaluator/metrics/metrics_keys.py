from enum import Enum

class MetricKey(str, Enum):
    """
    Enum for metric keys used in the evaluation process.
    Each metric key corresponds to a specific evaluation metric.
    """
    CONTEXT_PRECISION = "context_precision"
    ASPECT_CRITIC = "aspect_critic"
    ASPECT_CRITIC_MALICIOUSNESS = "aspect_critic_maliciousness"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
