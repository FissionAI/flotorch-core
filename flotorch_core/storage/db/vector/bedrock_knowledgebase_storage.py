import boto3
from typing import List, Dict, Any, Optional
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.storage.db.vector.vector_storage import VectorStorage, VectorStorageSearchItem, VectorStorageSearchResponse
from flotorch_core.embedding.embedding import BaseEmbedding, EmbeddingMetadata


logger = get_logger()


class BedrockKnowledgeBaseStorage(VectorStorage):
    def __init__(self, knowledge_base_id: str, region: str = 'us-east-1', embedder: Optional[BaseEmbedding] = None):
        self.client = boto3.client("bedrock-agent-runtime", region_name=region)
        self.knowledge_base_id = knowledge_base_id

    def search(self, chunk, knn: int, hierarchical: bool = False) -> VectorStorageSearchResponse:
        """
        Searches the Bedrock Knowledge Base using vector search.
        """
        query = {"text": chunk.data}
        retrieval_configuration = {
            'vectorSearchConfiguration': {'numberOfResults': knn}
        }

        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery=query,
                retrievalConfiguration=retrieval_configuration
            )
            formatted_results = self._format_response(response)

            return VectorStorageSearchResponse(
                status=True,
                result=formatted_results,
                metadata={
                    "embedding_metadata": EmbeddingMetadata(0, 0)
                }
            )

        except Exception as e:
            logger.error(f"Error retrieving from Bedrock Knowledge Base: {str(e)}")
            return VectorStorageSearchResponse(
                status=False,
                result=[],
                metadata={"error": str(e)}
            )

    def _format_response(self, data) -> List[VectorStorageSearchItem]:
        formatted_results = []
        for result in data.get('retrievalResults', []):
            content = result.get('content', {})
            text = content.get('text', '')

            if text:
                formatted_results.append(VectorStorageSearchItem(text=text))

        return formatted_results

    def embed_query(self, query_vector: List[float], knn: int, hierarchical: bool = False) -> Dict[str, Any]:
        """
        Bedrock Knowledge Base does not support explicit query vectorization, as embedding is managed internally.
        """
        raise NotImplementedError("Embedding is managed internally by Bedrock Knowledge Base.")

    def read(self, key) -> dict:
        raise NotImplementedError("Not implemented")

    def write(self, item: dict):
        raise NotImplementedError("Not implemented.")