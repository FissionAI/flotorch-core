from flotorch_core.storage.db.vector.no_ops_vector_storage import NoOpsVectorStorage
from flotorch_core.storage.db.vector.open_search import OpenSearchClient
from flotorch_core.storage.db.vector.bedrock_knowledgebase_storage import BedrockKnowledgeBaseStorage
from flotorch_core.storage.db.vector.vector_storage import VectorStorage
from flotorch_core.embedding.embedding import BaseEmbedding
from typing import Optional

class VectorStorageFactory:
    @staticmethod
    def create_vector_storage(
        knowledge_base: bool,
        use_bedrock_kb: bool,
        embedding: BaseEmbedding,
        opensearch_host: Optional[str] = None,
        opensearch_port: Optional[int] = None,
        opensearch_username: Optional[str] = None,
        opensearch_password: Optional[str] = None,
        index_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        aws_region: str = "us-east-1"
    ) -> VectorStorage:
        """
        Factory method to return the appropriate vector storage client.

        :param use_bedrock_kb: Boolean flag to decide which storage to use.
        :param embedding: The embedding model to use.
        :param opensearch_host: OpenSearch host (Only needed for OpenSearch).
        :param opensearch_port: OpenSearch port (Only needed for OpenSearch).
        :param opensearch_username: OpenSearch username (Only needed for OpenSearch).
        :param opensearch_password: OpenSearch password (Only needed for OpenSearch).
        :param index_id: OpenSearch index ID (Only needed for OpenSearch).
        :param knowledge_base_id: Bedrock Knowledge Base ID (Only needed for Bedrock KB).
        :param aws_region: AWS region for Bedrock Knowledge Base (Defaults to "us-east-1").
        :return: An instance of `VectorStorage` (either `OpenSearchClient` or `BedrockKnowledgeBaseStorage`).
        """
        if not knowledge_base:
            return NoOpsVectorStorage()

        if use_bedrock_kb:
            if not knowledge_base_id:
                raise ValueError("Knowledge Base ID must be provided when using Bedrock Knowledge Base.")
            
            return BedrockKnowledgeBaseStorage(
                knowledge_base_id=knowledge_base_id,
                region=aws_region,
                embedder=embedding
            )
        
        if not (opensearch_host and opensearch_port and opensearch_username and opensearch_password and index_id):
            raise ValueError("All OpenSearch parameters must be provided when using OpenSearch.")

        return OpenSearchClient(
            opensearch_host, 
            opensearch_port,
            opensearch_username, 
            opensearch_password,
            index_id,
            embedder=embedding
        )
