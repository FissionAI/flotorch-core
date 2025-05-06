import pytest
from unittest.mock import Mock, patch
import numpy as np
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.evaluator.evaluation_item import EvaluationItem
from flotorch_core.evaluator.metrics.metrics_keys import MetricKey
from flotorch_core.evaluator.ragas_evaluator import RagasEvaluator

@pytest.fixture
def mock_inferencer():
    mock = Mock()
    mock.generate_text = Mock(return_value=({"metadata": {}}, "generated answer"))
    return mock

@pytest.fixture
def mock_embedding():
    mock = Mock()
    mock.embed = Mock(return_value=Mock(embeddings=[np.array([0.1, 0.2, 0.3])]))
    return mock

@pytest.fixture
def ragas_evaluator(mock_inferencer, mock_embedding):
    with patch('flotorch_core.evaluator.ragas_evaluator.LangchainEmbeddingsWrapper') as mock_wrapper:
        evaluator = RagasEvaluator(
            evaluator_llm=mock_inferencer,
            embedding_llm=mock_embedding
        )
        return evaluator  # Return just the evaluator, not the tuple

def test_embedding_functionality(mock_inferencer, mock_embedding):
    """Test the embedding functionality through the wrapped embeddings"""
    with patch('flotorch_core.evaluator.ragas_evaluator.LangchainEmbeddingsWrapper') as mock_wrapper:
        evaluator = RagasEvaluator(
            evaluator_llm=mock_inferencer,
            embedding_llm=mock_embedding
        )
        
        mock_wrapper.assert_called_once()
        wrapper_args = mock_wrapper.call_args[0][0]
        
        test_text = "test text"
        
        # Configure mock_embedding to return a specific response
        mock_embedding_response = Mock()
        mock_embedding_response.embeddings = [[1.0, 2.0, 3.0]]
        mock_embedding.embed.return_value = mock_embedding_response
        
        # Test embed_documents through the wrapper
        result = wrapper_args.embed_documents([test_text])
        
        # Verify that the underlying embedding was called with a Chunk containing the correct data
        mock_embedding.embed.assert_called()  # First verify it was called
        actual_chunk = mock_embedding.embed.call_args[0][0]
        assert isinstance(actual_chunk, Chunk)
        assert actual_chunk.data == test_text  # Compare the content instead of the object
        
        # Verify the result matches the mock response
        assert result == [[1.0, 2.0, 3.0]]
        
        # Reset the mock for the next test
        mock_embedding.reset_mock()
        
        # Test embed_query
        query_result = wrapper_args.embed_query(test_text)
        
        # Verify the query embedding call
        mock_embedding.embed.assert_called()
        actual_chunk = mock_embedding.embed.call_args[0][0]
        assert isinstance(actual_chunk, Chunk)
        assert actual_chunk.data == test_text
        assert query_result == [1.0, 2.0, 3.0]

def test_llm_functionality(mock_inferencer, mock_embedding):
    """Test the LLM functionality through the wrapped LLM"""
    with patch('flotorch_core.evaluator.ragas_evaluator.LangchainLLMWrapper') as mock_wrapper:
        evaluator = RagasEvaluator(
            evaluator_llm=mock_inferencer,
            embedding_llm=mock_embedding
        )
        
        # Verify that LangchainLLMWrapper was called with an LLM wrapper
        mock_wrapper.assert_called_once()
        
        # Get the wrapper instance that was passed to LangchainLLMWrapper
        wrapper_class = mock_wrapper.call_args[0][0]
        
        # Test invoke
        result = wrapper_class.invoke("test prompt")
        assert result == "generated answer"
        mock_inferencer.generate_text.assert_called_with(
            user_query="test prompt",
            context=[]
        )
        
        # Reset mock for next test
        mock_inferencer.reset_mock()
        
        # Test generate_prompt
        result = wrapper_class.generate_prompt(["test prompt"])
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "generated answer"
        mock_inferencer.generate_text.assert_called_with(
            user_query="test prompt",
            context=[]
        )


@pytest.mark.asyncio
def test_llm_functionality(mock_inferencer, mock_embedding):
    """Test the LLM functionality through the wrapped LLM"""
    # Create a simple class to mimic prompts with text attribute
    class PromptLike:
        def __init__(self, text):
            self.text = text

    with patch('flotorch_core.evaluator.ragas_evaluator.LangchainLLMWrapper') as mock_wrapper:
        evaluator = RagasEvaluator(
            evaluator_llm=mock_inferencer,
            embedding_llm=mock_embedding
        )
        
        # Verify that LangchainLLMWrapper was called with an LLM wrapper
        mock_wrapper.assert_called_once()
        
        # Get the wrapper instance that was passed to LangchainLLMWrapper
        wrapper_class = mock_wrapper.call_args[0][0]
        
        # Test invoke
        result = wrapper_class.invoke("test prompt")
        assert result == "generated answer"
        mock_inferencer.generate_text.assert_called_with(
            user_query="test prompt",
            context=[]
        )
        
        # Reset mock for next test
        mock_inferencer.reset_mock()
        
        # Test generate_prompt with PromptLike objects
        test_prompts = [PromptLike("test prompt")]
        result = wrapper_class.generate_prompt(test_prompts)
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "generated answer"
        mock_inferencer.generate_text.assert_called_with(
            user_query="test prompt",
            context=[]
        )

def test_initialization(mock_inferencer, mock_embedding):
    """Test basic initialization of RagasEvaluator"""
    evaluator = RagasEvaluator(
        evaluator_llm=mock_inferencer,
        embedding_llm=mock_embedding
    )
    
    assert evaluator.evaluator_llm == mock_inferencer
    assert evaluator.embedding_llm == mock_embedding
    assert evaluator.metric_args is None

def test_initialization_with_metric_args(mock_inferencer, mock_embedding):
    """Test initialization with metric arguments"""
    metric_args = {
        MetricKey.ASPECT_CRITIC: {
            "test_aspect": {
                "name": "test",
                "definition": "test definition"
            }
        }
    }
    
    evaluator = RagasEvaluator(
        evaluator_llm=mock_inferencer,
        embedding_llm=mock_embedding,
        metric_args=metric_args
    )
    
    assert evaluator.metric_args == metric_args

def test_embedding_wrapper(mock_inferencer, mock_embedding):
    """Test the embedded _EmbeddingWrapper class"""
    with patch('flotorch_core.evaluator.ragas_evaluator.LangchainEmbeddingsWrapper') as mock_wrapper:
        evaluator = RagasEvaluator(
            evaluator_llm=mock_inferencer,
            embedding_llm=mock_embedding
        )
        
        # Get the wrapper instance that was passed to LangchainEmbeddingsWrapper
        mock_wrapper.assert_called_once()
        wrapper = mock_wrapper.call_args[0][0]
        
        # Configure mock_embedding to return a specific response
        mock_embedding_response = Mock()
        mock_embedding_response.embeddings = [[1.0, 2.0, 3.0]]
        mock_embedding.embed.return_value = mock_embedding_response
        
        # Test embed_documents
        result = wrapper.embed_documents(["test text"])
        assert len(result) == 1
        assert len(result[0]) == 3  # Based on our mock embedding size
        
        # Verify the embedding call
        mock_embedding.embed.assert_called()  # First verify it was called
        actual_chunk = mock_embedding.embed.call_args[0][0]
        assert isinstance(actual_chunk, Chunk)
        assert actual_chunk.data == "test text"  # Compare the content instead of the object
        
        # Reset mock for next test
        mock_embedding.reset_mock()
        
        # Test embed_query
        result = wrapper.embed_query("test query")
        assert len(result) == 3
        
        # Verify the query embedding call
        mock_embedding.embed.assert_called()
        actual_chunk = mock_embedding.embed.call_args[0][0]
        assert isinstance(actual_chunk, Chunk)
        assert actual_chunk.data == "test query"

@pytest.mark.asyncio
async def test_llm_wrapper_async(mock_inferencer, mock_embedding):
    """Test the embedded _LLMWrapper class async methods"""
    evaluator = RagasEvaluator(
        evaluator_llm=mock_inferencer,
        embedding_llm=mock_embedding
    )
    
    wrapper = evaluator._LLMWrapper(mock_inferencer)
    
    # Test ainvoke
    result = await wrapper.ainvoke("test prompt")
    assert result == "generated answer"
    
    # Test agenerate_prompt
    result = await wrapper.agenerate_prompt(["test prompt"])
    assert len(result.generations) == 1
    assert result.generations[0][0].text == "generated answer"

def test_llm_wrapper_sync(mock_inferencer, mock_embedding):
    """Test the embedded _LLMWrapper class sync methods"""
    with patch('flotorch_core.evaluator.ragas_evaluator.LangchainLLMWrapper') as mock_wrapper:
        evaluator = RagasEvaluator(
            evaluator_llm=mock_inferencer,
            embedding_llm=mock_embedding
        )
        
        # Verify that LangchainLLMWrapper was called
        mock_wrapper.assert_called_once()
        
        # Get the wrapper instance that was passed to LangchainLLMWrapper
        wrapper = mock_wrapper.call_args[0][0]
        
        # Test invoke
        result = wrapper.invoke("test prompt")
        assert result == "generated answer"
        mock_inferencer.generate_text.assert_called_with(
            user_query="test prompt",
            context=[]
        )
        
        # Reset mock for next test
        mock_inferencer.reset_mock()
        
        # Test generate_prompt
        class PromptLike:
            def __init__(self, text):
                self.text = text
                
        test_prompts = [PromptLike("test prompt")]
        result = wrapper.generate_prompt(test_prompts)
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "generated answer"
        mock_inferencer.generate_text.assert_called_with(
            user_query="test prompt",
            context=[]
        )

def test_evaluate_with_specific_metrics(ragas_evaluator):
    """Test evaluation with specific metrics"""
    evaluation_items = [
        EvaluationItem(
            question="test question",
            generated_answer="test answer",
            expected_answer="expected answer"
        )
    ]
    
    specific_metrics = [MetricKey.FAITHFULNESS]
    
    with patch('flotorch_core.evaluator.ragas_evaluator.evaluate') as mock_evaluate:
        mock_evaluate.return_value = {"faithfulness": 0.9}
        
        result = ragas_evaluator.evaluate(evaluation_items, metrics=specific_metrics)
        
        assert result == {"faithfulness": 0.9}
        mock_evaluate.assert_called_once()



def test_evaluate_with_specific_metrics(ragas_evaluator):
    """Test evaluation with specific metrics"""
    evaluation_items = [
        EvaluationItem(
            question="test question",
            generated_answer="test answer",
            expected_answer="expected answer"
        )
    ]
    
    specific_metrics = [MetricKey.FAITHFULNESS]
    
    with patch('flotorch_core.evaluator.ragas_evaluator.evaluate') as mock_evaluate:
        mock_evaluate.return_value = {"faithfulness": 0.9}
        
        result = ragas_evaluator.evaluate(evaluation_items, metrics=specific_metrics)
        
        assert result == {"faithfulness": 0.9}
        mock_evaluate.assert_called_once()


def test_evaluate_empty_data(ragas_evaluator):
    """Test evaluation with empty data"""
    with patch('flotorch_core.evaluator.ragas_evaluator.evaluate') as mock_evaluate:
        result = ragas_evaluator.evaluate([])
        mock_evaluate.assert_called_once()

@pytest.mark.asyncio
async def test_concurrent_evaluation(ragas_evaluator):
    """Test concurrent evaluation requests"""
    evaluation_items = [
        EvaluationItem(
            question=f"question_{i}",
            generated_answer=f"answer_{i}",
            expected_answer=f"expected_{i}"
        )
        for i in range(3)
    ]
    
    with patch('flotorch_core.evaluator.ragas_evaluator.evaluate') as mock_evaluate:
        mock_evaluate.return_value = {"score": 0.8}
        
        # Simulate concurrent evaluations
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(ragas_evaluator.evaluate, evaluation_items)
            )
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert all(result == {"score": 0.8} for result in results)
        assert mock_evaluate.call_count == 3
