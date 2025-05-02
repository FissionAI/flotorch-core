from flotorch_core.storage.db.vector.vector_storage import VectorStorage
from flotorch_core.rerank.rerank import BedrockReranker

class RerankedVectorStorage(VectorStorage):
    """
    A class that combines a vector storage with a reranker.
    It first retrieves the top-k results from the vector storage and then reranks them using a reranker.
    """

    def __init__(self, vectorStorage: VectorStorage, bedrockReranker: BedrockReranker):
        """
        Initializes the RerankedVectorStorage with a vector storage and a reranker.
        Args:
            vectorStorage (VectorStorage): The vector storage to use for initial retrieval.
            bedrockReranker (BedrockReranker): The reranker to use for reranking the results.
        """
        self.vectorStorage = vectorStorage
        self.reranker = bedrockReranker
    
    def search(self, query: str, knn: int, hierarchical=False):
        """
        Searches for the top-k results in the vector storage and reranks them using the reranker.
        Args:
            query (str): The query to search for.
            knn (int): The number of nearest neighbors to retrieve.
            hierarchical (bool): Whether to use hierarchical search or not.
        Returns:
            list: The reranked results.
        """
        results = self.vectorStorage.search(query, knn, hierarchical)
        return self.reranker.rerank_documents(query, results)

    def embed_query(self, embedding, knn, hierarical=False):
        """
        Embeds the query using the vector storage and reranks the results.
        Args:
            embedding: The embedding to use for the query.
            knn (int): The number of nearest neighbors to retrieve.
            hierarical (bool): Whether to use hierarchical search or not."""
        self.vectorStorage.embed_query(embedding, knn, hierarical)