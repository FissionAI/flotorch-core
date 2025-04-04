from aiohttp import ClientError
import pytest
from flotorch_core.embedding.titanv2_embedding import TitanV2Embedding
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.logger.global_logger import get_logger

logger = get_logger()

@pytest.fixture
def embedding():
    return TitanV2Embedding(
        model_id="amazon.titan-embed-text-v2:0",
        region="us-east-1",
        dimensions=256,
        normalize=True
    )

def test_titanv2_initialization():
    # Test default initialization
    embedder = TitanV2Embedding(model_id="amazon.titan-embed-text-v2:0")
    assert embedder.model_id == "amazon.titan-embed-text-v2:0"
    assert embedder.region == "us-east-1"  # default region
    assert embedder.dimension == 256  # default dimension
    assert embedder.normalize is True  # default normalize

    # Test custom initialization
    embedder = TitanV2Embedding(
        model_id="amazon.titan-embed-text-v2:0",
        region="us-west-2",
        dimensions=512,
        normalize=False
    )
    assert embedder.model_id == "amazon.titan-embed-text-v2:0"
    assert embedder.region == "us-west-2"
    assert embedder.dimension == 512
    assert embedder.normalize is False

def test_prepare_chunk(embedding):
    # Test chunk preparation
    chunk = Chunk(data="Test text")
    prepared_data = embedding._prepare_chunk(chunk)
    
    assert isinstance(prepared_data, dict)
    assert "inputText" in prepared_data
    assert "dimensions" in prepared_data
    assert "normalize" in prepared_data
    
    assert prepared_data["inputText"] == "Test text"
    assert prepared_data["dimensions"] == 256
    assert prepared_data["normalize"] is True

def test_prepare_chunk_with_different_settings():
    embedder = TitanV2Embedding(
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=512,
        normalize=False
    )
    
    chunk = Chunk(data="Test text")
    prepared_data = embedder._prepare_chunk(chunk)
    
    assert prepared_data["dimensions"] == 512
    assert prepared_data["normalize"] is False
    assert prepared_data["inputText"] == "Test text"

def test_prepare_chunk_with_empty_text():
    embedder = TitanV2Embedding(model_id="amazon.titan-embed-text-v2:0")
    chunk = Chunk(data="")
    prepared_data = embedder._prepare_chunk(chunk)
    
    assert prepared_data["inputText"] == ""
    assert prepared_data["dimensions"] == 256
    assert prepared_data["normalize"] is True

def test_prepare_chunk_with_special_characters():
    embedder = TitanV2Embedding(model_id="amazon.titan-embed-text-v2:0")
    chunk = Chunk(data="Special chars: !@#$%^&*()")
    prepared_data = embedder._prepare_chunk(chunk)
    
    assert prepared_data["inputText"] == "Special chars: !@#$%^&*()"

def test_prepare_chunk_with_unicode():
    embedder = TitanV2Embedding(model_id="amazon.titan-embed-text-v2:0")
    chunk = Chunk(data="Unicode text: 你好世界")
    prepared_data = embedder._prepare_chunk(chunk)
    
    assert prepared_data["inputText"] == "Unicode text: 你好世界"

def test_prepare_chunk_with_long_text():
    embedder = TitanV2Embedding(model_id="amazon.titan-embed-text-v2:0")
    long_text = "This is a very long text. " * 100
    chunk = Chunk(data=long_text)
    prepared_data = embedder._prepare_chunk(chunk)
    
    assert prepared_data["inputText"] == long_text
    assert prepared_data["dimensions"] == 256
    assert prepared_data["normalize"] is True

def test_prepare_chunk_with_newlines():
    embedder = TitanV2Embedding(model_id="amazon.titan-embed-text-v2:0")
    text_with_newlines = "Line 1\nLine 2\nLine 3"
    chunk = Chunk(data=text_with_newlines)
    prepared_data = embedder._prepare_chunk(chunk)
    
    assert prepared_data["inputText"] == text_with_newlines

def test_prepare_chunk_with_long_text():
    embedder = TitanV2Embedding(model_id="amazon.titan-embed-text-v2:0")
    long_text = "This is a very long text. " * 100
    chunk = Chunk(data=long_text)
    prepared_data = embedder._prepare_chunk(chunk)
    
    assert prepared_data["inputText"] == long_text
    assert prepared_data["dimensions"] == 256
    assert prepared_data["normalize"] is True

def test_prepare_chunk_with_newlines():
    embedder = TitanV2Embedding(model_id="amazon.titan-embed-text-v2:0")
    text_with_newlines = "Line 1\nLine 2\nLine 3"
    chunk = Chunk(data=text_with_newlines)
    prepared_data = embedder._prepare_chunk(chunk)
    
    assert prepared_data["inputText"] == text_with_newlines
