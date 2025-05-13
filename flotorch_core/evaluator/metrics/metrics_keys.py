from enum import Enum

class MetricKey(str, Enum):
    CONTEXT_PRECISION = "context_precision"
    ASPECT_CRITIC = "aspect_critic"
    ASPECT_CRITIC_MALICIOUSNESS = "aspect_critic_maliciousness"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    SEMANTIC_SIMILARITY="semantic_similarity"
