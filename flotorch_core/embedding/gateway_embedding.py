"""
This class is responsible for embedding the text using the Gateway model.
"""
from typing import Dict
from openai import OpenAI
from flotorch_core.embedding.embedding import BaseEmbedding
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.embedding.embedding import Embeddings, EmbeddingMetadata
from flotorch_core.embedding.embedding_registry import register


@register("gateway")
class GatewayEmbedding(BaseEmbedding):
    """
    Initializes the GatewayEmbedding class.
    :param model_id: The model id of the Gateway model.
    :param base_url: The base url of the console.
    :param api_key: The api key of the console.
    :param headers: The headers of the console.
    :param dimensions: The dimensions of the embedding.
    :param normalize: Normalize the embeddings.
    """
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        headers: Dict[str, str] = None,
        dimensions: int = 256,
        normalize: bool = True,
    ):
        super().__init__(model_id, None, dimensions, normalize)
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, default_headers=self.headers
        )

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"input": chunk.data}

    def embed(self, chunk: Chunk) -> Embeddings:
        response = self.client.embeddings.create(input=chunk.data, model=self.model_id)
        metadata = EmbeddingMetadata(
            input_tokens=response.usage.total_tokens, latency_ms=0.0
        )
        return Embeddings(
            embeddings=response.data[0].embedding, metadata=metadata, text=chunk.data
        )
