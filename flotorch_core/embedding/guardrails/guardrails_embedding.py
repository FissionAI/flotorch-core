from flotorch_core.embedding.embedding import BaseEmbedding
from typing import List, Dict
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.embedding.embedding import Embeddings, EmbeddingList
from flotorch_core.guardrails.guardrails import BaseGuardRail


class GuardrailsEmbedding(BaseEmbedding):
    """
    This class is a wrapper for embedding models that applies guardrails to the input text before embedding.
    It inherits from the BaseEmbedding class and implements the embed and embed_list methods.
    """

    def __init__(self, base_embedding: BaseEmbedding, 
                 base_guardrail: BaseGuardRail):
        """
        Initializes the GuardrailsEmbedding class.
        Args:
            base_embedding (BaseEmbedding): The base embedding model to be used.
            base_guardrail (BaseGuardRail): The guardrail to be applied to the input text.
        Returns:
            None
        """
        super().__init__(base_embedding.dimension, base_embedding.normalize)
        self.base_embedding = base_embedding
        self.base_guardrail = base_guardrail

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        """
        Prepares the chunk for embedding by applying the guardrail.
        Args:
            chunk (Chunk): The chunk to be prepared.
        Returns:
            Dict: The prepared chunk.
        """
        return self.base_embedding._prepare_chunk(chunk)

    
    def embed(self, chunk: Chunk) -> Embeddings:
        """
        Embeds the chunk.
        Args:
            chunk (Chunk): The chunk to be embedded.
        Returns:
            Embeddings: The embedding of the chunk.
        """
        guardrail_response = self.base_guardrail.apply_guardrail(text=chunk.data)
        if guardrail_response['action'] == 'GUARDRAIL_INTERVENED':
            return None

        return self.base_embedding.embed(chunk)

    def embed_list(self, chunks: List[Chunk]) -> EmbeddingList:
        """
        Embeds the list of chunks.
        Args:
            chunks (List[Chunk]): The list of chunks to be embedded.
        Returns:
            EmbeddingList: The list of embeddings.
        """
        embedding_list = EmbeddingList()
        if not isinstance(chunks, list):
            return embedding_list.append(self.embed(chunks))
        for chunk in chunks:
            embedding = self.embed(chunk)
            if not embedding is None:
                embedding_list.append(embedding)
        return embedding_list
