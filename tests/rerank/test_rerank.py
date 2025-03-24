import pytest
from unittest.mock import Mock, patch
from flotorch_core.rerank.rerank import BedrockReranker

@pytest.fixture
def mock_bedrock_client():
    """Creates a mock Bedrock client."""
    return Mock()

def test_bedrock_reranker_initialization_with_mock_client(mock_bedrock_client):
    """Tests initialization when a mock Bedrock client is provided."""
    reranker = BedrockReranker(region="us-west-2", rerank_model_id="test-model", bedrock_client=mock_bedrock_client)
    assert reranker.region == "us-west-2"
    assert reranker.rerank_model_id == "test-model"
    assert reranker.bedrock_agent_runtime == mock_bedrock_client

@patch("flotorch_core.rerank.rerank.boto3.client")
def test_bedrock_reranker_initialization_without_mock_client(mock_boto3_client):
    """Tests initialization when no client is provided (boto3 should be called)."""
    mock_boto3_client.return_value = Mock()  # Mock boto3 client return
    reranker = BedrockReranker(region="us-west-2", rerank_model_id="test-model")

    mock_boto3_client.assert_called_once_with("bedrock-agent-runtime", region_name="us-west-2")
    assert reranker.rerank_model_id == "test-model"

def test_rerank_documents_empty_list(mock_bedrock_client):
    """Tests reranking with an empty document list (should return empty)."""
    reranker = BedrockReranker(region="us-west-2", rerank_model_id="test-model", bedrock_client=mock_bedrock_client)
    result = reranker.rerank_documents("query", [])
    assert result == []

def test_rerank_documents_success(mock_bedrock_client):
    """Tests reranking with valid inputs and mocked API response."""
    reranker = BedrockReranker(region="us-west-2", rerank_model_id="test-model", bedrock_client=mock_bedrock_client)

    input_prompt = "Find relevant docs"
    retrieved_documents = [{"text": "Doc1"}, {"text": "Doc2"}, {"text": "Doc3"}]

    # Mock response from Bedrock API
    mock_bedrock_client.rerank.return_value = {
        "results": [{"index": 2}, {"index": 0}, {"index": 1}]
    }

    result = reranker.rerank_documents(input_prompt, retrieved_documents)

    expected_result = [{"text": "Doc3"}, {"text": "Doc1"}, {"text": "Doc2"}]
    assert result == expected_result
    mock_bedrock_client.rerank.assert_called_once()

def test_rerank_documents_no_results(mock_bedrock_client):
    """Tests when Bedrock returns no results (should return empty)."""
    reranker = BedrockReranker(region="us-west-2", rerank_model_id="test-model", bedrock_client=mock_bedrock_client)
    mock_bedrock_client.rerank.return_value = {"results": []}

    result = reranker.rerank_documents("query", [{"text": "Doc1"}])
    assert result == []

def test_rerank_documents_exception_handling(mock_bedrock_client):
    """Tests error handling when Bedrock API raises an exception."""
    reranker = BedrockReranker(region="us-west-2", rerank_model_id="test-model", bedrock_client=mock_bedrock_client)
    mock_bedrock_client.rerank.side_effect = Exception("Bedrock API error")

    result = reranker.rerank_documents("query", [{"text": "Doc1"}])
    assert result == []
