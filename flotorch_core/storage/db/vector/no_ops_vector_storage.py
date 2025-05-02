import boto3
from typing import List, Dict, Any
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.storage.db.vector.vector_storage import VectorStorage, VectorStorageSearchItem, VectorStorageSearchResponse
from flotorch_core.embedding.embedding import EmbeddingMetadata


logger = get_logger()


class NoOpsVectorStorage(VectorStorage):
    """
    NoOpsVectorStorage is a placeholder for vector storage that does not perform any operations.
    It is used when no knowledge base is required.
    """
    def search(self, chunk, knn: int, hierarchical: bool = False) -> VectorStorageSearchResponse:
        """
        used when no knowledgebase is required
        Args:
            chunk: The chunk to search for.
            knn: The number of nearest neighbors to return.
            hierarchical: Whether to use hierarchical search.
        Returns:
            VectorStorageSearchResponse: The search response.
        """
        return VectorStorageSearchResponse(
            status=True,
            result=[],
            metadata={
                "embedding_metadata": EmbeddingMetadata(0, 0)
            }
        )

    def embed_query(self, query_vector: List[float], knn: int, hierarchical: bool = False) -> Dict[str, Any]:
        """
        Bedrock Knowledge Base does not support explicit query vectorization, as embedding is managed internally.
        Args:
            query_vector: The query vector to embed.
            knn: The number of nearest neighbors to return.
            hierarchical: Whether to use hierarchical search.
        Returns:
            Dict[str, Any]: The embedding metadata.
        """
        raise NotImplementedError("Embedding is managed internally by Bedrock Knowledge Base.")

    def read(self, key) -> dict:
        """
        Read a document from the vector storage.
        Args:
            key: The key of the document to read.
        Returns:
            dict: The document.
        """
        raise NotImplementedError("Not implemented")

    def write(self, item: dict):
        """
        Write a document to the vector storage.
        Args:
            item: The document to write.
        """
        raise NotImplementedError("Not implemented.")