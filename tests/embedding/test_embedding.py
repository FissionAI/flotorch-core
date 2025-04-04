import pytest
from flotorch_core.embedding.embedding import EmbeddingMetadata, Embeddings, EmbeddingList, BaseEmbedding
from flotorch_core.chunking.chunking import Chunk

# Fixtures
@pytest.fixture
def sample_metadata():
    return EmbeddingMetadata(input_tokens=100, latency_ms=50)

@pytest.fixture
def sample_embeddings():
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

@pytest.fixture
def sample_text():
    return "Sample text for testing"

# Test functions
def test_embedding_metadata_initialization(sample_metadata):
    assert sample_metadata.input_tokens == 100
    assert sample_metadata.latency_ms == 50

def test_embedding_metadata_append(sample_metadata):
    additional_metadata = EmbeddingMetadata(input_tokens=50, latency_ms=25)
    sample_metadata.append(additional_metadata)
    assert sample_metadata.input_tokens == 150
    assert sample_metadata.latency_ms == 75

def test_embedding_metadata_to_json(sample_metadata):
    json_data = sample_metadata.to_json()
    assert isinstance(json_data, dict)
    assert json_data['input_token'] == 100
    assert json_data['latency_ms'] == 50

@pytest.fixture
def sample_embedding(sample_embeddings, sample_metadata, sample_text):
    return Embeddings(
        embeddings=sample_embeddings,
        metadata=sample_metadata,
        text=sample_text
    )

def test_embeddings_initialization(sample_embedding, sample_embeddings, sample_metadata, sample_text):
    assert sample_embedding.embeddings == sample_embeddings
    assert sample_embedding.metadata == sample_metadata
    assert sample_embedding.text == sample_text
    assert sample_embedding.id == ''

@pytest.mark.parametrize("input_text, expected_output", [
    ("Simple text", "Simple text"),
    ("Text with 'quotes'", "Text with quotes"),
    ('Text with "quotes"', "Text with quotes"),
    ("Text with\nnewlines", "Text with newlines"),
    ("Text with\ttabs", "Text with tabs"),
    ("Text with @#$% symbols", "Text with symbols"),
    ("  Extra  spaces  ", "Extra spaces"),
    ("Mixed\nCase\tText!", "Mixed Case Text")
])
def test_clean_text_for_vector_db(sample_embedding, input_text, expected_output):
    cleaned_text = sample_embedding.clean_text_for_vector_db(input_text)
    assert cleaned_text == expected_output

def test_embeddings_to_json(sample_embedding):
    json_data = sample_embedding.to_json()
    assert isinstance(json_data, dict)
    assert json_data["vectors"] == sample_embedding.embeddings
    assert isinstance(json_data["text"], str)
    assert json_data["metadata"]["inputTokens"] == sample_embedding.metadata.input_tokens
    assert json_data["metadata"]["latencyMs"] == sample_embedding.metadata.latency_ms

@pytest.fixture
def empty_embedding_list():
    return EmbeddingList()

def test_embedding_list_initialization(empty_embedding_list):
    assert empty_embedding_list.embeddings == []
    assert isinstance(empty_embedding_list.metadata, EmbeddingMetadata)
    assert empty_embedding_list.metadata.input_tokens == 0
    assert empty_embedding_list.metadata.latency_ms == 0

def test_embedding_list_append(empty_embedding_list, sample_embeddings, sample_text):
    embedding1 = Embeddings(
        embeddings=sample_embeddings,
        metadata=EmbeddingMetadata(input_tokens=100, latency_ms=50),
        text=sample_text
    )
    
    embedding2 = Embeddings(
        embeddings=sample_embeddings,
        metadata=EmbeddingMetadata(input_tokens=200, latency_ms=75),
        text=sample_text
    )
    
    empty_embedding_list.append(embedding1)
    assert len(empty_embedding_list.embeddings) == 1
    assert empty_embedding_list.metadata.input_tokens == 100
    assert empty_embedding_list.metadata.latency_ms == 50
    
    empty_embedding_list.append(embedding2)
    assert len(empty_embedding_list.embeddings) == 2
    assert empty_embedding_list.metadata.input_tokens == 300
    assert empty_embedding_list.metadata.latency_ms == 125

# Test BaseEmbedding
class ConcreteEmbedding(BaseEmbedding):
    def _prepare_chunk(self, chunk):
        return {"text": chunk.data}
    
    def embed(self, chunk):
        return Embeddings(
            embeddings=[[0.1, 0.2]],
            metadata=EmbeddingMetadata(input_tokens=100, latency_ms=50),
            text=chunk.data
        )

@pytest.fixture
def concrete_embedding():
    return ConcreteEmbedding(
        model_id="test-model",
        region="us-east-1"
    )

def test_base_embedding_initialization(concrete_embedding):
    assert concrete_embedding.model_id == "test-model"
    assert concrete_embedding.region == "us-east-1"
    assert concrete_embedding.dimension == 256
    assert concrete_embedding.normalize is True

def test_base_embedding_single_chunk(concrete_embedding):
    chunk = Chunk(data="Test chunk")
    result = concrete_embedding.embed_list(chunk)
    assert isinstance(result, EmbeddingList)
    assert len(result.embeddings) == 1
    assert result.metadata.input_tokens == 100
    assert result.metadata.latency_ms == 50

def test_base_embedding_multiple_chunks(concrete_embedding):
    chunk1 = Chunk(data="Chunk 1")
    chunk2 = Chunk(data="Chunk 2")
    
    # Create child chunks and add them to chunk2
    child1 = Chunk(data="Child 1")
    child2 = Chunk(data="Child 2")
    chunk2.add_child(child1)
    chunk2.add_child(child2)
    
    chunks = [chunk1, chunk2]
    result = concrete_embedding.embed_list(chunks)
    assert isinstance(result, EmbeddingList)
    assert len(result.embeddings) == 3
    assert result.metadata.input_tokens == 300
    assert result.metadata.latency_ms == 150

@pytest.mark.parametrize("dimensions,normalize", [
    (512, False),
    (1024, True),
    (128, False)
])
def test_custom_parameters(dimensions, normalize):
    embedding = ConcreteEmbedding(
        model_id="test-model",
        region="us-east-1",
        dimensions=dimensions,
        normalize=normalize
    )
    assert embedding.dimension == dimensions
    assert embedding.normalize == normalize
