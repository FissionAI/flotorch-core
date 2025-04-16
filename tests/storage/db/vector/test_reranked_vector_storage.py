import pytest
from flotorch_core.storage.db.vector.reranked_vector_storage import RerankedVectorStorage
from flotorch_core.storage.db.vector.vector_storage import (
    VectorStorage,
    VectorStorageSearchItem,
    VectorStorageSearchResponse
)
from flotorch_core.rerank.rerank import BedrockReranker
from typing import List

# Mock classes for testing
class MockVectorStorage(VectorStorage):
    def __init__(self):
        self.storage = {}
    
    def read(self, key: str) -> dict:
        return self.storage.get(key, {})
    
    def write(self, item: dict):
        if 'id' in item:
            self.storage[item['id']] = item
    
    def search(self, query: str, knn: int, hierarchical=False) -> VectorStorageSearchResponse:
        return VectorStorageSearchResponse(
            status=True,
            result=[
                VectorStorageSearchItem(
                    text="Original result 1",
                    vectors=[0.1, 0.2, 0.3],
                    metadata={"score": 0.8}
                ),
                VectorStorageSearchItem(
                    text="Original result 2",
                    vectors=[0.4, 0.5, 0.6],
                    metadata={"score": 0.6}
                )
            ]
        )
    
    def embed_query(self, embedding: List[float], knn: int, hierarical=False) -> VectorStorageSearchResponse:
        return VectorStorageSearchResponse(
            status=True,
            result=[
                VectorStorageSearchItem(
                    text="Embedded result",
                    vectors=embedding
                )
            ]
        )

class MockBedrockReranker(BedrockReranker):
    def __init__(self, model_id: str = "test-model", region: str = "us-east-1"):
        super().__init__(rerank_model_id=model_id, region=region)
    
    def rerank_documents(self, query: str, documents: VectorStorageSearchResponse) -> VectorStorageSearchResponse:
        # Simulate reranking by reversing the order and modifying scores
        reranked_results = documents.result.copy()
        reranked_results.reverse()
        
        for i, result in enumerate(reranked_results):
            result.metadata["reranked_score"] = 1.0 - (i * 0.2)
        
        return VectorStorageSearchResponse(
            status=True,
            result=reranked_results,
            metadata={"reranked": True}
        )

@pytest.fixture
def vector_storage():
    return MockVectorStorage()

@pytest.fixture
def bedrock_reranker():
    return MockBedrockReranker(model_id="test-model", region="us-east-1")

@pytest.fixture
def reranked_storage(vector_storage, bedrock_reranker):
    return RerankedVectorStorage(vector_storage, bedrock_reranker)

class TestRerankedVectorStorage:
    def test_initialization(self, vector_storage, bedrock_reranker):
        """Test proper initialization of RerankedVectorStorage"""
        storage = RerankedVectorStorage(vector_storage, bedrock_reranker)
        
        assert storage.vectorStorage == vector_storage
        assert storage.reranker == bedrock_reranker

    def test_search_with_reranking(self, reranked_storage):
        """Test search method with reranking"""
        query = "test query"
        results = reranked_storage.search(query, knn=2)
        
        assert isinstance(results, VectorStorageSearchResponse)
        assert results.status is True
        assert len(results.result) == 2
        assert results.metadata.get("reranked") is True
        
        # Check if results are reranked (reversed in our mock implementation)
        assert results.result[0].text == "Original result 2"
        assert results.result[1].text == "Original result 1"
        
        # Check reranking scores
        assert results.result[0].metadata["reranked_score"] == 1.0
        assert results.result[1].metadata["reranked_score"] == 0.8

    def test_search_with_hierarchical(self, reranked_storage):
        """Test search method with hierarchical flag"""
        query = "test query"
        results = reranked_storage.search(query, knn=2, hierarchical=True)
        
        assert isinstance(results, VectorStorageSearchResponse)
        assert results.status is True
        assert len(results.result) == 2
        assert results.metadata.get("reranked") is True

    def test_embed_query(self, reranked_storage):
        """Test embed_query method"""
        embedding = [0.1, 0.2, 0.3]
        results = reranked_storage.embed_query(embedding, knn=2)
        
        assert isinstance(results, VectorStorageSearchResponse)
        assert results.status is True
        assert len(results.result) > 0
        assert results.result[0].vectors == embedding

    def test_embed_query_with_hierarchical(self, reranked_storage):
        """Test embed_query method with hierarchical flag"""
        embedding = [0.1, 0.2, 0.3]
        results = reranked_storage.embed_query(embedding, knn=2, hierarical=True)
        
        assert isinstance(results, VectorStorageSearchResponse)
        assert results.status is True
        assert len(results.result) > 0
        assert results.result[0].vectors == embedding

    def test_search_empty_results(self, vector_storage, bedrock_reranker):
        """Test search with empty results"""
        # Create a mock storage that returns empty results
        class EmptyMockVectorStorage(MockVectorStorage):
            def search(self, query: str, knn: int, hierarchical=False):
                return VectorStorageSearchResponse(status=True, result=[])
        
        storage = RerankedVectorStorage(EmptyMockVectorStorage(), bedrock_reranker)
        results = storage.search("test query", knn=2)
        
        assert isinstance(results, VectorStorageSearchResponse)
        assert results.status is True
        assert len(results.result) == 0

    def test_search_preserves_metadata(self, vector_storage, bedrock_reranker):
        """Test that search preserves original metadata while adding reranking information"""
        # Create a mock storage with metadata
        class MetadataMockVectorStorage(MockVectorStorage):
            def search(self, query: str, knn: int, hierarchical=False):
                return VectorStorageSearchResponse(
                    status=True,
                    result=[
                        VectorStorageSearchItem(
                            text="Test result",
                            vectors=[0.1, 0.2, 0.3],
                            metadata={"original_score": 0.9, "source": "test"}
                        )
                    ],
                    metadata={"original_metadata": True}
                )
        
        storage = RerankedVectorStorage(MetadataMockVectorStorage(), bedrock_reranker)
        results = storage.search("test query", knn=1)
        
        assert results.status is True
        assert len(results.result) == 1
        assert results.result[0].metadata.get("original_score") == 0.9
        assert results.result[0].metadata.get("source") == "test"
        assert results.result[0].metadata.get("reranked_score") is not None
        assert results.metadata.get("reranked") is True
