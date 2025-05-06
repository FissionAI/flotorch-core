import pytest
from typing import List
from flotorch_core.evaluator.metrics.base_metrics import BaseEvaluationMetric

def test_cannot_instantiate_abstract_class():
    """Test that BaseEvaluationMetric cannot be instantiated directly"""
    with pytest.raises(TypeError) as exc_info:
        BaseEvaluationMetric()
    assert "Can't instantiate abstract class" in str(exc_info.value)

def test_must_implement_abstract_methods():
    """Test that concrete classes must implement all abstract methods"""
    # Create a class that inherits but doesn't implement all methods
    class IncompleteMetric(BaseEvaluationMetric):
        @classmethod
        def available_metrics(cls) -> List[str]:
            return []
        # Missing other abstract methods

    with pytest.raises(TypeError) as exc_info:
        IncompleteMetric()
    assert "Can't instantiate abstract class" in str(exc_info.value)

def test_concrete_implementation():
    """Test that a concrete implementation works correctly"""
    class ConcreteMetric(BaseEvaluationMetric):
        _metrics = ["metric1", "metric2"]
        
        @classmethod
        def available_metrics(cls) -> List[str]:
            return cls._metrics

        @classmethod
        def get_metric(cls, key: str):
            if key in cls._metrics:
                return f"Result for {key}"
            raise ValueError(f"Unknown metric: {key}")

        @classmethod
        def initialize_metrics(cls, llm, embeddings):
            cls._llm = llm
            cls._embeddings = embeddings

    # Test the concrete implementation
    assert ConcreteMetric.available_metrics() == ["metric1", "metric2"]
    assert ConcreteMetric.get_metric("metric1") == "Result for metric1"
    
    # Test invalid metric key
    with pytest.raises(ValueError):
        ConcreteMetric.get_metric("invalid_metric")

    # Test initialization
    mock_llm = "mock_llm"
    mock_embeddings = "mock_embeddings"
    ConcreteMetric.initialize_metrics(mock_llm, mock_embeddings)
    assert ConcreteMetric._llm == mock_llm
    assert ConcreteMetric._embeddings == mock_embeddings

def test_method_signatures():
    """Test that method signatures match the abstract class"""
    class TestMetric(BaseEvaluationMetric):
        @classmethod
        def available_metrics(cls) -> List[str]:
            return []

        @classmethod
        def get_metric(cls, key: str):
            pass

        @classmethod
        def initialize_metrics(cls, llm, embeddings):
            pass

    # Verify return type annotation for available_metrics
    from typing import get_type_hints
    return_type = get_type_hints(TestMetric.available_metrics)['return']
    assert return_type == List[str]

    # Verify method parameters
    from inspect import signature
    init_params = signature(TestMetric.initialize_metrics).parameters
    assert 'llm' in init_params
    assert 'embeddings' in init_params
