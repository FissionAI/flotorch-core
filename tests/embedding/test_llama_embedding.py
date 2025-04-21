import pytest
from unittest.mock import patch
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.embedding.llama_embedding import LlamaEmbedding
from flotorch_core.embedding.embedding import Embeddings

@pytest.fixture
def llama_embedding():
    """Fixture to initialize LlamaEmbedding instance."""
    return LlamaEmbedding(model_id="llama2:7b", region="us-west-2", dimensions=256, normalize=True)

@pytest.fixture
def test_chunk():
    """Fixture to create a sample chunk for testing."""
    return Chunk("This is a sample text for embedding.")

@pytest.fixture
def sample_embedding():
    """Fixture to provide a sample embedding response."""
    return [0.1] * 256

def test_init(llama_embedding):
    """Test initialization of LlamaEmbedding class."""
    assert llama_embedding.model_id == "llama2:7b"
    assert llama_embedding.region == "us-west-2"
    assert llama_embedding.dimension == 256
    assert llama_embedding.normalize is True

@patch('ollama.embeddings')
def test_embed(mock_ollama_embeddings, llama_embedding, test_chunk, sample_embedding):
    """Test embed method of LlamaEmbedding class."""
    # Setup mock response
    mock_ollama_embeddings.return_value = {'embedding': sample_embedding}

    # Call the embed method
    result = llama_embedding.embed(test_chunk)

    # Create expected embedding object with the correct text
    expected_embedding = Embeddings(
        embeddings=sample_embedding,
        text=test_chunk.data,  # This is the text from the chunk
        metadata=None
    )
    
    # Verify the result
    assert isinstance(result, Embeddings)
    assert result.embeddings == expected_embedding.embeddings
    assert result.text == test_chunk.data  # Compare directly with chunk data
    assert result.metadata is None
    
    # Verify the mock was called correctly
    mock_ollama_embeddings.assert_called_once_with(
        model="llama2:7b",
        prompt=test_chunk.data
    )

@patch('ollama.embeddings')
def test_error_handling(mock_ollama_embeddings, llama_embedding, test_chunk):
    """Test error handling in embed method."""
    mock_ollama_embeddings.side_effect = Exception("API Error")
    
    with pytest.raises(Exception) as exc_info:
        llama_embedding.embed(test_chunk)
    assert str(exc_info.value) == "API Error"
