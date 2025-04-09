import pytest
from flotorch_core.storage.db.vector.vector_storage import (
    VectorStorageSearchItem,
    VectorStorageSearchResponse,
    VectorStorage
)
from flotorch_core.embedding.embedding import BaseEmbedding
from typing import List

# Mock class for testing - with all required parameters and methods
class MockEmbeddingForTest(BaseEmbedding):
    def __init__(self, model_id: str = "test-model", region: str = "us-east-1", return_vectors: List[float] = None):
        super().__init__(model_id=model_id, region=region)
        self.return_vectors = return_vectors or [0.1, 0.2, 0.3]

    def embed(self, text: str) -> List[float]:
        return self.return_vectors
    
    def _prepare_chunk(self, text: str) -> str:
        return text  # Simple implementation for testing
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.return_vectors for _ in texts]

# Helper class for testing - with all required methods
class ConcreteStorageForTest(VectorStorage):
    def __init__(self, embedder=None):
        super().__init__(embedder=embedder)
        self.storage = {}  # Simple in-memory storage for testing

    def read(self, key: str) -> dict:
        """Implement required read method"""
        return self.storage.get(key, {})

    def write(self, item: dict):
        """Implement required write method"""
        if 'id' in item:
            self.storage[item['id']] = item

    def search(self, query: str, knn, hierarchical=False):
        return VectorStorageSearchResponse(
            status=True,
            result=[
                VectorStorageSearchItem(
                    text=query,
                    vectors=[0.1, 0.2, 0.3]
                )
            ]
        )

    def embed_query(self, embedding, knn, hierarical=False):
        return VectorStorageSearchResponse(
            status=True,
            result=[
                VectorStorageSearchItem(
                    text="test",
                    vectors=embedding
                )
            ]
        )

@pytest.fixture
def mock_embedder():
    return MockEmbeddingForTest(model_id="test-model", region="us-east-1")


class TestVectorStorageSearchItem:
    def test_creation_with_all_fields(self):
        """Test VectorStorageSearchItem creation with all fields"""
        item = VectorStorageSearchItem(
            text="test text",
            execution_id="exec123",
            chunk_id="chunk123",
            parent_id="parent123",
            vectors=[0.1, 0.2, 0.3],
            metadata={"key": "value"}
        )

        assert item.text == "test text"
        assert item.execution_id == "exec123"
        assert item.chunk_id == "chunk123"
        assert item.parent_id == "parent123"
        assert item.vectors == [0.1, 0.2, 0.3]
        assert item.metadata == {"key": "value"}

    def test_creation_with_minimal_fields(self):
        """Test VectorStorageSearchItem creation with minimal fields"""
        item = VectorStorageSearchItem(text="test text")

        assert item.text == "test text"
        assert item.execution_id is None
        assert item.chunk_id is None
        assert item.parent_id is None
        assert item.vectors == []
        assert item.metadata == {}

    def test_to_json_conversion(self):
        """Test VectorStorageSearchItem to_json method"""
        item = VectorStorageSearchItem(
            text="test text",
            execution_id="exec123",
            chunk_id="chunk123",
            parent_id="parent123",
            vectors=[0.1, 0.2, 0.3],
            metadata={"key": "value"}
        )

        json_data = item.to_json()
        
        assert json_data == {
            "text": "test text",
            "execution_id": "exec123",
            "chunk_id": "chunk123",
            "parent_id": "parent123",
            "vectors": [0.1, 0.2, 0.3],
            "metadata": {"key": "value"}
        }

class TestVectorStorageSearchResponse:
    def test_creation_with_all_fields(self):
        """Test VectorStorageSearchResponse creation with all fields"""
        item = VectorStorageSearchItem(text="test text")
        response = VectorStorageSearchResponse(
            status=True,
            result=[item],
            metadata={"total": 1}
        )

        assert response.status is True
        assert len(response.result) == 1
        assert response.result[0] == item
        assert response.metadata == {"total": 1}

    def test_creation_with_minimal_fields(self):
        """Test VectorStorageSearchResponse creation with minimal fields"""
        response = VectorStorageSearchResponse(status=True)

        assert response.status is True
        assert response.result == []
        assert response.metadata == {}

    def test_to_json_conversion(self):
        """Test VectorStorageSearchResponse to_json method"""
        item = VectorStorageSearchItem(
            text="test text",
            vectors=[0.1, 0.2, 0.3]
        )
        response = VectorStorageSearchResponse(
            status=True,
            result=[item],
            metadata={"total": 1}
        )

        json_data = response.to_json()
        
        assert json_data == {
            "status": True,
            "result": [item.to_json()],
            "metadata": {"total": 1}
        }

class TestVectorStorageBase:
    def test_initialization_with_embedder(self, mock_embedder):
        """Test VectorStorage initialization with embedder"""
        storage = ConcreteStorageForTest(embedder=mock_embedder)
        assert storage.embedder == mock_embedder
    
    def test_initialization_with_custom_embedder(self):
        """Test VectorStorage initialization with custom embedder"""
        custom_vectors = [0.5, 0.6, 0.7]
        embedder = MockEmbeddingForTest(
            model_id="test-model",
            region="us-east-1",
            return_vectors=custom_vectors
        )
        storage = ConcreteStorageForTest(embedder=embedder)
        
        assert storage.embedder == embedder
        assert storage.embedder.embed("test") == custom_vectors

    def test_initialization_without_embedder(self):
        """Test VectorStorage initialization without embedder"""
        storage = ConcreteStorageForTest()
        assert storage.embedder is None

    def test_abstract_class_instantiation(self):
        """Test that VectorStorage cannot be instantiated directly"""
        with pytest.raises(TypeError) as exc_info:
            VectorStorage()
        assert "Can't instantiate abstract class VectorStorage" in str(exc_info.value)

    def test_abstract_methods_definition(self):
        """Test that abstract methods are properly defined"""
        abstract_methods = VectorStorage.__abstractmethods__
        assert 'search' in abstract_methods
        assert 'embed_query' in abstract_methods

class TestVectorStorageImplementation:
    def test_concrete_search_method(self):
        """Test concrete implementation of search method"""
        storage = ConcreteStorageForTest()
        search_response = storage.search("test query", knn=5)
        
        assert isinstance(search_response, VectorStorageSearchResponse)
        assert search_response.status is True
        assert len(search_response.result) > 0
        assert isinstance(search_response.result[0], VectorStorageSearchItem)

    def test_concrete_embed_query_method(self):
        """Test concrete implementation of embed_query method"""
        storage = ConcreteStorageForTest()
        embed_response = storage.embed_query([0.1, 0.2, 0.3], knn=5)
        
        assert isinstance(embed_response, VectorStorageSearchResponse)
        assert embed_response.status is True
        assert len(embed_response.result) > 0
        assert isinstance(embed_response.result[0], VectorStorageSearchItem)

    def test_read_write_operations(self):
        """Test basic read/write operations"""
        storage = ConcreteStorageForTest()
        test_item = {"id": "test1", "data": "test data"}
        
        # Test write
        storage.write(test_item)
        
        # Test read
        result = storage.read("test1")
        assert result == test_item
        
        # Test reading non-existent key
        empty_result = storage.read("non_existent")
        assert empty_result == {}

    def test_hierarchical_search(self):
        """Test search with hierarchical flag"""
        storage = ConcreteStorageForTest()
        response = storage.search("test query", knn=5, hierarchical=True)
        
        assert isinstance(response, VectorStorageSearchResponse)
        assert response.status is True

    def test_hierarchical_embed_query(self):
        """Test embed_query with hierarchical flag"""
        storage = ConcreteStorageForTest()
        response = storage.embed_query([0.1, 0.2, 0.3], knn=5, hierarical=True)
        
        assert isinstance(response, VectorStorageSearchResponse)
        assert response.status is True