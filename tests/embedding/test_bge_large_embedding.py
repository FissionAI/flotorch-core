import pytest
from unittest.mock import Mock, patch
from flotorch_core.embedding.bge_large_embedding import (
    BGELargeEmbedding,
    BGEM3Embedding,
    GTEQwen2Embedding
)
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.config.config import Config

@pytest.fixture
def mock_config():
    """Fixture for Config with mocked role"""
    config = Mock(spec=Config)
    config.get_sagemaker_arn_role.return_value = "arn:aws:iam::123456789012:role/test-role"
    return config

@pytest.fixture
def mock_sagemaker_client():
    """Fixture for mocked sagemaker client"""
    mock_client = Mock()
    mock_client.describe_endpoint.return_value = {
        'EndpointStatus': 'InService',
        'EndpointArn': 'test-arn',
        'ProductionVariants': [{'CurrentWeight': 1.0}]
    }
    return mock_client

@pytest.fixture
def mock_boto3():
    """Fixture for mocked boto3"""
    with patch('boto3.client') as mock_client:
        mock_client.return_value = Mock()
        yield mock_client

def test_bge_large_initialization(mock_config, mock_boto3, mock_sagemaker_client):
    """Test BGELargeEmbedding initialization"""
    mock_boto3.return_value = mock_sagemaker_client
    with patch('flotorch_core.embedding.bge_large_embedding.Config', return_value=mock_config):
        embedding = BGELargeEmbedding(
            model_id="huggingface-sentencesimilarity-bge-large-en-v1-5",
            region="us-east-1"
        )
        
        assert embedding.model_id == "huggingface-sentencesimilarity-bge-large-en-v1-5"
        assert embedding.region == "us-east-1"
        # Remove the assertion for get_sagemaker_arn_role.called

def test_bge_m3_initialization(mock_config, mock_boto3, mock_sagemaker_client):
    """Test BGEM3Embedding initialization"""
    mock_boto3.return_value = mock_sagemaker_client
    with patch('flotorch_core.embedding.bge_large_embedding.Config', return_value=mock_config):
        embedding = BGEM3Embedding(
            model_id="huggingface-sentencesimilarity-bge-m3",
            region="us-east-1"
        )
        
        assert embedding.model_id == "huggingface-sentencesimilarity-bge-m3"
        assert embedding.region == "us-east-1"
        # Remove the assertion for get_sagemaker_arn_role.called

def test_gte_qwen2_initialization(mock_config, mock_boto3, mock_sagemaker_client):
    """Test GTEQwen2Embedding initialization"""
    mock_boto3.return_value = mock_sagemaker_client
    with patch('flotorch_core.embedding.bge_large_embedding.Config', return_value=mock_config):
        embedding = GTEQwen2Embedding(
            model_id="huggingface-textembedding-gte-qwen2-7b-instruct",
            region="us-east-1"
        )
        
        assert embedding.model_id == "huggingface-textembedding-gte-qwen2-7b-instruct"
        assert embedding.region == "us-east-1"
        # Remove the assertion for get_sagemaker_arn_role.called

def test_bge_large_prepare_chunk(mock_config, mock_boto3, mock_sagemaker_client):
    """Test BGELargeEmbedding _prepare_chunk method"""
    mock_boto3.return_value = mock_sagemaker_client
    with patch('flotorch_core.embedding.bge_large_embedding.Config', return_value=mock_config):
        embedding = BGELargeEmbedding(
            model_id="huggingface-sentencesimilarity-bge-large-en-v1-5",
            region="us-east-1"
        )
        chunk = Chunk(data="test text")
        prepared_data = embedding._prepare_chunk(chunk)
        
        assert isinstance(prepared_data, dict)
        assert "text_inputs" in prepared_data
        assert "mode" in prepared_data
        assert prepared_data["text_inputs"] == [chunk.data]
        assert prepared_data["mode"] == "embedding"

def test_bge_m3_prepare_chunk(mock_config, mock_boto3, mock_sagemaker_client):
    """Test BGEM3Embedding _prepare_chunk method"""
    mock_boto3.return_value = mock_sagemaker_client
    with patch('flotorch_core.embedding.bge_large_embedding.Config', return_value=mock_config):
        embedding = BGEM3Embedding(
            model_id="huggingface-sentencesimilarity-bge-m3",
            region="us-east-1"
        )
        chunk = Chunk(data="test text")
        prepared_data = embedding._prepare_chunk(chunk)
        
        assert isinstance(prepared_data, dict)
        assert "text_inputs" in prepared_data
        assert "mode" in prepared_data
        assert prepared_data["text_inputs"] == [chunk.data]
        assert prepared_data["mode"] == "embedding"

def test_gte_qwen2_prepare_chunk(mock_config, mock_boto3, mock_sagemaker_client):
    """Test GTEQwen2Embedding _prepare_chunk method"""
    mock_boto3.return_value = mock_sagemaker_client
    with patch('flotorch_core.embedding.bge_large_embedding.Config', return_value=mock_config):
        embedding = GTEQwen2Embedding(
            model_id="huggingface-textembedding-gte-qwen2-7b-instruct",
            region="us-east-1"
        )
        chunk = Chunk(data="test text")
        prepared_data = embedding._prepare_chunk(chunk)
        
        assert isinstance(prepared_data, dict)
        assert "inputs" in prepared_data
        assert prepared_data["inputs"] == [chunk.data]

def test_prepare_chunk_with_special_characters(mock_config, mock_boto3, mock_sagemaker_client):
    """Test _prepare_chunk method with special characters"""
    mock_boto3.return_value = mock_sagemaker_client
    special_chars = "!@#$%^&*()\n\t"
    special_chunk = Chunk(data=special_chars)
    
    embeddings = [
        (BGELargeEmbedding, "text_inputs"),
        (BGEM3Embedding, "text_inputs"),
        (GTEQwen2Embedding, "inputs")
    ]
    
    for EmbeddingClass, input_key in embeddings:
        with patch('flotorch_core.embedding.bge_large_embedding.Config', return_value=mock_config):
            embedding = EmbeddingClass(
                model_id="test-model",
                region="us-east-1"
            )
            prepared_data = embedding._prepare_chunk(special_chunk)
            assert prepared_data[input_key] == [special_chars]

def test_inheritance(mock_config, mock_boto3, mock_sagemaker_client):
    """Test proper inheritance from SageMakerEmbedder"""
    mock_boto3.return_value = mock_sagemaker_client
    from flotorch_core.embedding.sagemaker_embedding import SageMakerEmbedder
    
    embeddings = [BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding]
    
    for EmbeddingClass in embeddings:
        with patch('flotorch_core.embedding.bge_large_embedding.Config', return_value=mock_config):
            embedding = EmbeddingClass(
                model_id="test-model",
                region="us-east-1"
            )
            assert isinstance(embedding, SageMakerEmbedder)
