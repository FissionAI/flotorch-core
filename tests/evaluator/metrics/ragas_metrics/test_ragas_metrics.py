import pytest
from unittest.mock import MagicMock
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, AspectCritic
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey
from flotorch_core.evaluator.metrics.ragas_metrics.ragas_metrics import RagasEvaluationMetrics


@pytest.fixture
def mock_llm_and_embeddings():
    # Create mock LLM and embeddings objects
    llm = MagicMock()
    embeddings = MagicMock()
    return llm, embeddings


def test_initialize_metrics_without_args(mock_llm_and_embeddings):
    llm, embeddings = mock_llm_and_embeddings

    # Initialize metrics without passing any metric_args
    RagasEvaluationMetrics.initialize_metrics(llm, embeddings)

    # Assert that metrics are initialized for the available keys
    assert "context_precision" in RagasEvaluationMetrics._initialized_metrics
    assert "faithfulness" in RagasEvaluationMetrics._initialized_metrics
    assert "aspect_critic" in RagasEvaluationMetrics._initialized_metrics
    assert "answer_relevance" in RagasEvaluationMetrics._initialized_metrics

    # Check that default metrics are initialized
    assert "default" in RagasEvaluationMetrics._initialized_metrics["faithfulness"]
    assert isinstance(RagasEvaluationMetrics._initialized_metrics["faithfulness"]["default"], Faithfulness)


def test_initialize_metrics_with_args(mock_llm_and_embeddings):
    llm, embeddings = mock_llm_and_embeddings
    metric_args = {
        MetricKey.ASPECT_CRITIC: {
            "maliciousness": {
                "name": "maliciousness",
                "definition": "Is the response harmful?"
            }
        }
    }

    # Initialize metrics with metric_args
    RagasEvaluationMetrics.initialize_metrics(llm, embeddings, metric_args)

    # Check if the specific instance is initialized
    assert "aspect_critic" in RagasEvaluationMetrics._initialized_metrics
    assert "maliciousness" in RagasEvaluationMetrics._initialized_metrics["aspect_critic"]
    assert isinstance(RagasEvaluationMetrics._initialized_metrics["aspect_critic"]["maliciousness"], AspectCritic)


def test_initialize_metrics_missing_args(mock_llm_and_embeddings):
    llm, embeddings = mock_llm_and_embeddings
    metric_args = {
        MetricKey.ASPECT_CRITIC: {
            "maliciousness": {
                "name": "maliciousness",
                # Missing definition here
            }
        }
    }

    # Test that a ValueError is raised due to missing required args
    with pytest.raises(ValueError, match="Metric 'aspect_critic' is missing required args: definition"):
        RagasEvaluationMetrics.initialize_metrics(llm, embeddings, metric_args)


def test_available_metrics(mock_llm_and_embeddings):
    llm, embeddings = mock_llm_and_embeddings

    # Initialize metrics
    RagasEvaluationMetrics.initialize_metrics(llm, embeddings)

    # Test the available metrics method
    available = RagasEvaluationMetrics.available_metrics()
    assert "context_precision" in available
    assert "faithfulness" in available
    assert "aspect_critic" in available
    assert "answer_relevance" in available


def test_get_metric(mock_llm_and_embeddings):
    llm, embeddings = mock_llm_and_embeddings
    RagasEvaluationMetrics.initialize_metrics(llm, embeddings)

    # Test getting a valid metric
    metrics = RagasEvaluationMetrics.get_metric(MetricKey.FAITHFULNESS)
    assert "default" in metrics
    assert isinstance(metrics["default"], Faithfulness)

    # Test getting an invalid metric (should raise ValueError)
    with pytest.raises(ValueError, match="Metric 'invalid_metric' has not been initialized"):
        RagasEvaluationMetrics.get_metric("invalid_metric")
