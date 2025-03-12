from typing import List

import ollama
from .embedding import BaseEmbedding, Embeddings, EmbeddingMetadata
from flotorch_core.chunking.chunking import Chunk
from .embedding_registry import register

"""

This class is responsible for embedding the text using the Llama model.
If Ollama server is running remotely set the environment variable OLLAMA_HOST to the server URL."""
@register("llama2")
class LlamaEmbedding(BaseEmbedding):
    """
    Initializes the LlamaEmbedding class.
    :param model_id: The model id of the Llama model.
    """

    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True):
        super().__init__(model_id, region, dimensions, normalize)

    """
    Prepares the chunk for embedding.
    :param chunk: The chunk to be embedded.
    :return: The prepared chunk.
    """
    def embed(self, chunk: Chunk) -> Embeddings:
        response = ollama.embeddings(model=self.model_id, prompt=chunk.data)
        embedding = Embeddings(embeddings=response['embedding'], metadata=None)
        return embedding