from flotorch_core.embedding.embedding import BaseEmbedding
from typing import List, Dict
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.embedding.embedding import Embeddings, EmbeddingList
from flotorch_core.guardrails.guardrails import BaseGuardRail


class GuardrailsEmbedding(BaseEmbedding):

    def __init__(self, base_embedding: BaseEmbedding, 
                 base_guardrail: BaseGuardRail):
        super().__init__(base_embedding.dimension, base_embedding.normalize)
        self.base_embedding = base_embedding
        self.base_guardrail = base_guardrail

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return self.base_embedding._prepare_chunk(chunk)

    """
    Embeds the chunk.
    :param chunk: The chunk to be embedded.
    :return: The embeddings.
    """
    def embed(self, chunk: Chunk) -> Embeddings:
        guardrail_response = self.base_guardrail.apply_guardrail(text=chunk.data)
        if guardrail_response['action'] == 'GUARDRAIL_INTERVENED':
            return None

        return self.base_embedding.embed(chunk)

    """
    Embeds the list of chunks.
    :param chunks: The list of chunks to be embedded.
    :return: The list of embeddings.
    """
    def embed_list(self, chunks: List[Chunk]) -> EmbeddingList:
        embedding_list = EmbeddingList()
        if not isinstance(chunks, list):
            return embedding_list.append(self.embed(chunks))
        for chunk in chunks:
            embedding = self.embed(chunk)
            if not embedding is None:
                embedding_list.append(embedding)
        return embedding_list
