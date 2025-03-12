import boto3
from typing import List, Dict, Any
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.storage.db.vector.vector_storage import VectorStorage, VectorStorageSearchItem, VectorStorageSearchResponse
from flotorch_core.embedding.embedding import EmbeddingMetadata


logger = get_logger()


class NoOpsVectorStorage(VectorStorage):
    def search(self, chunk, knn: int, hierarchical: bool = False) -> VectorStorageSearchResponse:
        """
        used when no knowledgebase is required
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
        """
        raise NotImplementedError("Embedding is managed internally by Bedrock Knowledge Base.")

    def read(self, key) -> dict:
        raise NotImplementedError("Not implemented")

    def write(self, item: dict):
        raise NotImplementedError("Not implemented.")