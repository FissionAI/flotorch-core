import pytest
import json
from unittest.mock import Mock, patch
from flotorch_core.embedding.bedrock_embedding import BedRockEmbedding
from flotorch_core.chunking.chunking import Chunk
from typing import Dict, Any, List

class TestBedRockEmbedding(BedRockEmbedding):
    """Concrete test class for BedRockEmbedding"""
    def _prepare_chunk(self, chunk):
        return {"inputText": chunk.data}
        
    def embed_chunk(self, chunk: Chunk) -> list:
        """Embed a chunk of text"""
        prepared_data = self._prepare_chunk(chunk)
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(prepared_data)
        )
        response_body = json.loads(response['body'].read())
        return self.extract_embedding(response_body)

    def extract_embedding(self, response: Dict[str, Any]) -> List[float]:
        """Extract embedding from response"""
        if "embedding" not in response:
            raise KeyError("Response does not contain 'embedding' key")
        
        embedding = response["embedding"]
        if embedding is None:
            raise TypeError("Embedding cannot be None")
        if not isinstance(embedding, list):
            raise TypeError("Embedding must be a list")
            
        return embedding
    
@pytest.fixture
def mock_bedrock_client():
    """Fixture for mocked Bedrock client"""
    mock_client = Mock()
    mock_client.invoke_model.return_value = {
        'body': Mock(
            read=Mock(return_value=b'{"embedding": [0.1, 0.2, 0.3]}')
        )
    }
    return mock_client

@pytest.fixture
def mock_boto3():
    """Fixture for mocked boto3"""
    with patch('boto3.client') as mock_client:
        mock_client.return_value = Mock()
        yield mock_client

@pytest.fixture
def embedding(mock_boto3, mock_bedrock_client):
    """Fixture for TestBedRockEmbedding instance"""
    mock_boto3.return_value = mock_bedrock_client
    return TestBedRockEmbedding(
        model_id="amazon.titan-embed-text-v1",
        region="us-east-1"
    )

@pytest.fixture
def mock_chunk():
    """Fixture for test chunk"""
    return Chunk(data="test text")

def test_initialization(mock_boto3, mock_bedrock_client):
    """Test BedRockEmbedding initialization"""
    mock_boto3.return_value = mock_bedrock_client
    embedding = TestBedRockEmbedding(
        model_id="amazon.titan-embed-text-v1",
        region="us-east-1"
    )
    
    assert embedding.model_id == "amazon.titan-embed-text-v1"
    assert embedding.region == "us-east-1"

def test_extract_embedding(embedding):
    """Test extract_embedding method"""
    mock_response = {
        "embedding": [0.1, 0.2, 0.3]
    }
    
    result = embedding.extract_embedding(mock_response)
    
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]

def test_extract_embedding_empty_response(embedding):
    """Test extract_embedding method with empty response"""
    mock_response = {}
    
    with pytest.raises(KeyError, match="Response does not contain 'embedding' key"):
        embedding.extract_embedding(mock_response)

def test_extract_embedding_invalid_response(embedding):
    """Test extract_embedding method with invalid response"""
    test_cases = [
        ({"wrong_key": []}, KeyError, "Response does not contain 'embedding' key"),
        ({"embedding": None}, TypeError, "Embedding cannot be None"),
        ({"embedding": "not_a_list"}, TypeError, "Embedding must be a list"),
    ]
    
    for response, expected_error, expected_message in test_cases:
        with pytest.raises(expected_error, match=expected_message):
            embedding.extract_embedding(response)

def test_embed_chunk(embedding, mock_chunk, mock_bedrock_client):
    """Test embed_chunk method"""
    result = embedding.embed_chunk(mock_chunk)
    
    assert isinstance(result, list)
    assert len(result) > 0
    mock_bedrock_client.invoke_model.assert_called_once()

def test_embed_chunk_with_empty_text(embedding, mock_bedrock_client):
    """Test embed_chunk method with empty text"""
    empty_chunk = Chunk(data="")
    
    result = embedding.embed_chunk(empty_chunk)
    
    assert isinstance(result, list)
    mock_bedrock_client.invoke_model.assert_called_once()

def test_embed_chunk_client_error(mock_boto3):
    """Test embed_chunk method when client raises an error"""
    mock_client = Mock()
    mock_client.invoke_model.side_effect = Exception("API Error")
    mock_boto3.return_value = mock_client
    
    embedding = TestBedRockEmbedding(
        model_id="amazon.titan-embed-text-v1",
        region="us-east-1"
    )
    chunk = Chunk(data="test text")
    
    with pytest.raises(Exception):
        embedding.embed_chunk(chunk)

def test_embed_chunk_invalid_json_response(mock_boto3):
    """Test embed_chunk method with invalid JSON response"""
    mock_client = Mock()
    mock_client.invoke_model.return_value = {
        'body': Mock(
            read=Mock(return_value=b'invalid json')
        )
    }
    mock_boto3.return_value = mock_client
    
    embedding = TestBedRockEmbedding(
        model_id="amazon.titan-embed-text-v1",
        region="us-east-1"
    )
    chunk = Chunk(data="test text")
    
    with pytest.raises(Exception):
        embedding.embed_chunk(chunk)

def test_embed_chunk_large_text(embedding, mock_bedrock_client):
    """Test embed_chunk method with large text"""
    large_text = "a" * 10000  # 10K characters
    large_chunk = Chunk(data=large_text)
    
    result = embedding.embed_chunk(large_chunk)
    
    assert isinstance(result, list)
    mock_bedrock_client.invoke_model.assert_called_once()

def test_embed_chunk_unicode(embedding, mock_bedrock_client):
    """Test embed_chunk method with unicode characters"""
    unicode_chunk = Chunk(data="Hello 世界 🌍")
    
    result = embedding.embed_chunk(unicode_chunk)
    
    assert isinstance(result, list)
    mock_bedrock_client.invoke_model.assert_called_once()

def test_model_validation(mock_boto3, mock_bedrock_client):
    """Test model validation during initialization"""
    mock_boto3.return_value = mock_bedrock_client
    valid_models = [
        "amazon.titan-embed-text-v1",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3"
    ]
    
    for model_id in valid_models:
        embedding = TestBedRockEmbedding(
            model_id=model_id,
            region="us-east-1"
        )
        assert embedding.model_id == model_id

def test_region_validation(mock_boto3, mock_bedrock_client):
    """Test region validation during initialization"""
    mock_boto3.return_value = mock_bedrock_client
    valid_regions = ["us-east-1", "us-west-2", "eu-west-1"]
    
    for region in valid_regions:
        embedding = TestBedRockEmbedding(
            model_id="amazon.titan-embed-text-v1",
            region=region
        )
        assert embedding.region == region

def test_client_initialization(mock_boto3):
    """Test client initialization"""
    embedding = TestBedRockEmbedding(
        model_id="amazon.titan-embed-text-v1",
        region="us-east-1"
    )
    
    mock_boto3.assert_called_once_with(
        'bedrock-runtime',
        region_name='us-east-1'
    )
