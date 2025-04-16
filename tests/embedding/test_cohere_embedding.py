import pytest
from unittest.mock import Mock, patch
from flotorch_core.embedding.cohere_embedding import CohereEmbedding
from flotorch_core.chunking.chunking import Chunk

@pytest.fixture
def embedding():
    """Fixture for CohereEmbedding instance"""
    return CohereEmbedding(
        model_id="cohere.embed-multilingual-v3",
        region="us-east-1"
    )

@pytest.fixture
def mock_chunk():
    """Fixture for test chunk"""
    chunk = Chunk(data="test text")
    return chunk

def test_initialization():
    """Test CohereEmbedding initialization with different parameters"""
    # Test default parameters
    embedding = CohereEmbedding(
        model_id="cohere.embed-multilingual-v3",
        region="us-east-1"
    )
    assert embedding.model_id == "cohere.embed-multilingual-v3"
    assert embedding.region == "us-east-1"

    # Test with different model and region
    embedding = CohereEmbedding(
        model_id="cohere.embed-english-v3",
        region="us-west-2"
    )
    assert embedding.model_id == "cohere.embed-english-v3"
    assert embedding.region == "us-west-2"

def test_prepare_chunk(embedding, mock_chunk):
    """Test _prepare_chunk method"""
    prepared_data = embedding._prepare_chunk(mock_chunk)
    
    assert isinstance(prepared_data, dict)
    assert "texts" in prepared_data
    assert "input_type" in prepared_data
    assert prepared_data["texts"] == [mock_chunk.data]
    assert prepared_data["input_type"] == "search_document"

def test_extract_embedding(embedding):
    """Test extract_embedding method"""
    mock_response = {
        "embeddings": [
            [0.1, 0.2, 0.3, 0.4]
        ]
    }
    
    result = embedding.extract_embedding(mock_response)
    
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3, 0.4]

def test_extract_embedding_empty_response(embedding):
    """Test extract_embedding method with empty response"""
    with pytest.raises(IndexError):
        embedding.extract_embedding({"embeddings": []})


def test_extract_embedding_valid_response(embedding):
    """Test extract_embedding method with valid response"""
    mock_response = {
        "embeddings": [
            [0.1, 0.2, 0.3, 0.4]
        ]
    }
    
    result = embedding.extract_embedding(mock_response)
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3, 0.4]

def test_extract_embedding_none_response(embedding):
    """Test extract_embedding method with None response"""
    with pytest.raises(TypeError):
        embedding.extract_embedding(None)

def test_extract_embedding_invalid_response(embedding):
    """Test extract_embedding method with invalid response"""
    # Test empty dict
    with pytest.raises(KeyError):
        embedding.extract_embedding({})

    # Test wrong key
    with pytest.raises(KeyError):
        embedding.extract_embedding({"wrong_key": []})

    # Test None embeddings
    with pytest.raises(TypeError):
        embedding.extract_embedding({"embeddings": None})


@pytest.mark.parametrize("model_id", [
    "cohere.embed-multilingual-v3",
    "cohere.embed-english-v3"
])
def test_model_registration(model_id):
    """Test if models are properly registered"""
    embedding = CohereEmbedding(
        model_id=model_id,
        region="us-east-1"
    )
    assert embedding.model_id == model_id

def test_prepare_chunk_with_empty_data(embedding):
    """Test _prepare_chunk method with empty data"""
    empty_chunk = Chunk(data="")
    prepared_data = embedding._prepare_chunk(empty_chunk)
    
    assert prepared_data["texts"] == [""]
    assert prepared_data["input_type"] == "search_document"

def test_prepare_chunk_with_special_characters(embedding):
    """Test _prepare_chunk method with special characters"""
    special_chars_chunk = Chunk(data="!@#$%^&*()\n\t")
    prepared_data = embedding._prepare_chunk(special_chars_chunk)
    
    assert prepared_data["texts"] == ["!@#$%^&*()\n\t"]
    assert prepared_data["input_type"] == "search_document"

def test_inheritance():
    """Test proper inheritance from BedRockEmbedding"""
    embedding = CohereEmbedding(
        model_id="cohere.embed-multilingual-v3",
        region="us-east-1"
    )
    
    from flotorch_core.embedding.bedrock_embedding import BedRockEmbedding
    assert isinstance(embedding, BedRockEmbedding)

def test_prepare_chunk_large_text(embedding):
    """Test _prepare_chunk method with large text"""
    large_text = "a" * 10000  # 10K characters
    large_chunk = Chunk(data=large_text)
    prepared_data = embedding._prepare_chunk(large_chunk)
    
    assert len(prepared_data["texts"][0]) == 10000
    assert prepared_data["input_type"] == "search_document"

def test_prepare_chunk_unicode(embedding):
    """Test _prepare_chunk method with unicode characters"""
    unicode_chunk = Chunk(data="Hello 世界 🌍")
    prepared_data = embedding._prepare_chunk(unicode_chunk)
    
    assert prepared_data["texts"] == ["Hello 世界 🌍"]
    assert prepared_data["input_type"] == "search_document"
