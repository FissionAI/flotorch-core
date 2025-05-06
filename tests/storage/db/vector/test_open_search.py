import pytest
from unittest.mock import Mock, patch
from flotorch_core.storage.db.vector.open_search import OpenSearchClient
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.embedding.embedding import BaseEmbedding, Embeddings, EmbeddingMetadata
from opensearchpy import OpenSearch

class TestOpenSearchClient:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        # Create the main mock client
        self.mock_client = Mock(spec=OpenSearch)
        
        # Create and configure the indices mock
        self.mock_indices = Mock()
        self.mock_client.indices = self.mock_indices

        # Multiple patches to ensure no real connections are attempted
        patches = [
            patch('flotorch_core.storage.db.vector.open_search.OpenSearch', return_value=self.mock_client),
            patch('opensearchpy.OpenSearch', return_value=self.mock_client),
            patch('opensearchpy.client.OpenSearch', return_value=self.mock_client)
        ]
        
        # Start all patches
        for patcher in patches:
            patcher.start()
            
        yield
        
        # Stop all patches
        for patcher in patches:
            patcher.stop()

    @pytest.fixture
    def mock_embedder(self):
        embedder = Mock(spec=BaseEmbedding)
        embedder.embed.return_value = Embeddings(
            embeddings=[[0.1, 0.2, 0.3]],
            metadata=EmbeddingMetadata(input_tokens=3, latency_ms=100),
            text="test"
        )
        return embedder

    @pytest.fixture
    def client(self, mock_embedder, setup_mocks):
        return OpenSearchClient(
            host="localhost",
            port=9200,
            username="admin",
            password="admin",
            index="test-index",
            embedder=mock_embedder
        )

    def test_init(self, client):
        """Test initialization of OpenSearchClient"""
        assert client.host == "localhost"
        assert client.port == 9200
        assert client.username == "admin"
        assert client.password == "admin"
        assert client.index == "test-index"

    def test_write(self, client):
        """Test write method"""
        test_body = {"text": "test", "vectors": [0.1, 0.2, 0.3]}
        client.write(test_body)
        
        self.mock_client.index.assert_called_once_with(
            index="test-index",
            body=test_body
        )

    def test_write_bulk(self, client):
        """Test write_bulk method"""
        test_bodies = [
            {"text": "test1", "vectors": [0.1, 0.2, 0.3]},
            {"text": "test2", "vectors": [0.4, 0.5, 0.6]}
        ]
        client.write_bulk(test_bodies)
        
        self.mock_client.bulk.assert_called_once_with(body=test_bodies)

    def test_search(self, client):
        """Test search method"""
        # Mock the OpenSearch response
        mock_response = {
            'hits': {
                'hits': [{
                    '_id': '1',
                    '_source': {
                        'chunk_id': '123',
                        'parent_id': 'parent123',
                        'text': 'test text',
                        'vectors': [0.1, 0.2, 0.3],
                        'metadata': {'key': 'value'}
                    }
                }]
            }
        }
        self.mock_client.search.return_value = mock_response

        # Mock the mapping response
        mock_mapping = {
            'test-index': {
                'mappings': {
                    'properties': {
                        'vector_field': {'type': 'knn_vector'}
                    }
                }
            }
        }
        self.mock_indices.get_mapping.return_value = mock_mapping

        # Create test chunk with data parameter
        test_chunk = Chunk(data="test query")
        
        # Perform search
        result = client.search(test_chunk, knn=1)

        # Verify the response
        assert result.status is True
        assert len(result.result) == 1
        assert result.result[0].execution_id == '1'
        assert result.result[0].chunk_id == '123'
        assert result.result[0].parent_id == 'parent123'
        assert result.result[0].text == 'test text'
        assert result.result[0].vectors == [0.1, 0.2, 0.3]
        assert result.result[0].metadata == {'key': 'value'}

    def test_embed_query(self, client):
        """Test embed_query method"""
        # Mock the mapping response
        mock_mapping = {
            'test-index': {
                'mappings': {
                    'properties': {
                        'vector_field': {'type': 'knn_vector'}
                    }
                }
            }
        }
        self.mock_indices.get_mapping.return_value = mock_mapping

        query_vector = [0.1, 0.2, 0.3]
        result = client.embed_query(query_vector, knn=5)

        expected_query = {
            "size": 5,
            "query": {
                "knn": {
                    "vector_field": {
                        "vector": query_vector,
                        "k": 5
                    }
                }
            },
            "_source": True,
            "fields": ["text", "parent_id"]
        }

        assert result == expected_query
