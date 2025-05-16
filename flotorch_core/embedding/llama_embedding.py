from typing import List

import ollama
from .embedding import BaseEmbedding, Embeddings, EmbeddingMetadata
from flotorch_core.chunking.chunking import Chunk
from .embedding_registry import register


@register("llama2")
class LlamaEmbedding(BaseEmbedding):
    """
    This class is responsible for embedding the text using the Llama model.
    If Ollama server is running remotely set the environment variable OLLAMA_HOST to the server URL.
    """
    
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True):
        """
        Initializes the LlamaEmbedding class.
        Args:
            model_id (str): The ID of the model to be used for embedding.
            region (str): The region where the model is hosted.
            dimensions (int): The number of dimensions for the embedding. Default is 256.
            normalize (bool): Whether to normalize the embedding. Default is True.
        Returns:
            None
        """
        super().__init__(model_id, region, dimensions, normalize)

    
    def embed(self, chunk: Chunk) -> Embeddings:
        """
        Prepares the chunk for embedding.
        Args:
            chunk (Chunk): The chunk to be embedded.
        Returns:
            Embeddings: The embeddings for the chunk.
        """
        response = ollama.embeddings(model=self.model_id, prompt=chunk.data)
        embedding = Embeddings(embeddings=response['embedding'], metadata=None)
        return embedding