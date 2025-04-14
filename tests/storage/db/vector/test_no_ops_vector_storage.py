import pytest
from unittest.mock import Mock
from flotorch_core.storage.db.vector.no_ops_vector_storage import NoOpsVectorStorage
from flotorch_core.storage.db.vector.vector_storage import VectorStorageSearchResponse
from flotorch_core.embedding.embedding import EmbeddingMetadata
from flotorch_core.chunking.chunking import Chunk

class TestNoOpsVectorStorage:
    @pytest.fixture
    def storage(self):
        """Create a NoOpsVectorStorage instance for testing"""
        return NoOpsVectorStorage()

    def test_search(self, storage):
        """Test search method returns empty results with status True"""
        # Create a test chunk
        test_chunk = Chunk(data="test query")
        
        # Call search method
        response = storage.search(test_chunk, knn=5)
        
        # Verify response
        assert isinstance(response, VectorStorageSearchResponse)
        assert response.status is True
        assert response.result == []
        assert isinstance(response.metadata, dict)
        assert 'embedding_metadata' in response.metadata
        assert isinstance(response.metadata['embedding_metadata'], EmbeddingMetadata)
        assert response.metadata['embedding_metadata'].input_tokens == 0
        assert response.metadata['embedding_metadata'].latency_ms == 0

    def test_search_with_hierarchical(self, storage):
        """Test search method with hierarchical parameter"""
        test_chunk = Chunk(data="test query")
        
        # Call search method with hierarchical=True
        response = storage.search(test_chunk, knn=5, hierarchical=True)
        
        # Verify response is the same regardless of hierarchical parameter
        assert isinstance(response, VectorStorageSearchResponse)
        assert response.status is True
        assert response.result == []
        assert isinstance(response.metadata, dict)
        assert 'embedding_metadata' in response.metadata
        assert isinstance(response.metadata['embedding_metadata'], EmbeddingMetadata)
        assert response.metadata['embedding_metadata'].input_tokens == 0
        assert response.metadata['embedding_metadata'].latency_ms == 0

    def test_embed_query_raises_not_implemented(self, storage):
        """Test embed_query raises NotImplementedError"""
        test_vector = [0.1, 0.2, 0.3]
        
        with pytest.raises(NotImplementedError) as exc_info:
            storage.embed_query(test_vector, knn=5)
        
        assert str(exc_info.value) == "Embedding is managed internally by Bedrock Knowledge Base."

    def test_embed_query_with_hierarchical_raises_not_implemented(self, storage):
        """Test embed_query with hierarchical parameter raises NotImplementedError"""
        test_vector = [0.1, 0.2, 0.3]
        
        with pytest.raises(NotImplementedError) as exc_info:
            storage.embed_query(test_vector, knn=5, hierarchical=True)
        
        assert str(exc_info.value) == "Embedding is managed internally by Bedrock Knowledge Base."

    def test_read_raises_not_implemented(self, storage):
        """Test read method raises NotImplementedError"""
        with pytest.raises(NotImplementedError) as exc_info:
            storage.read("test_key")
        
        assert str(exc_info.value) == "Not implemented"

    def test_write_raises_not_implemented(self, storage):
        """Test write method raises NotImplementedError"""
        test_item = {"key": "value"}
        
        with pytest.raises(NotImplementedError) as exc_info:
            storage.write(test_item)
        
        assert str(exc_info.value) == "Not implemented."

    def test_search_with_none_chunk(self, storage):
        """Test search method with None chunk"""
        response = storage.search(None, knn=5)
        
        assert isinstance(response, VectorStorageSearchResponse)
        assert response.status is True
        assert response.result == []
        assert isinstance(response.metadata, dict)
        assert 'embedding_metadata' in response.metadata
        assert isinstance(response.metadata['embedding_metadata'], EmbeddingMetadata)

    def test_search_with_different_knn_values(self, storage):
        """Test search method with different knn values"""
        test_chunk = Chunk(data="test query")
        
        # Test with different knn values
        for knn in [1, 5, 10, 100]:
            response = storage.search(test_chunk, knn=knn)
            assert isinstance(response, VectorStorageSearchResponse)
            assert response.status is True
            assert response.result == []
            assert isinstance(response.metadata, dict)
            assert 'embedding_metadata' in response.metadata
            assert isinstance(response.metadata['embedding_metadata'], EmbeddingMetadata)

    def test_search_metadata_structure(self, storage):
        """Test detailed structure of search response metadata"""
        test_chunk = Chunk(data="test query")
        response = storage.search(test_chunk, knn=5)
        
        # Verify metadata structure
        assert isinstance(response.metadata, dict)
        assert len(response.metadata) == 1
        assert 'embedding_metadata' in response.metadata
        
        # Verify EmbeddingMetadata structure
        embedding_metadata = response.metadata['embedding_metadata']
        assert isinstance(embedding_metadata, EmbeddingMetadata)
        assert hasattr(embedding_metadata, 'input_tokens')
        assert hasattr(embedding_metadata, 'latency_ms')
        assert embedding_metadata.input_tokens == 0
        assert embedding_metadata.latency_ms == 0

    def test_write_with_none_item(self, storage):
        """Test write method with None item"""
        with pytest.raises(NotImplementedError) as exc_info:
            storage.write(None)
        
        assert str(exc_info.value) == "Not implemented."

    def test_read_with_none_key(self, storage):
        """Test read method with None key"""
        with pytest.raises(NotImplementedError) as exc_info:
            storage.read(None)
        
        assert str(exc_info.value) == "Not implemented"
