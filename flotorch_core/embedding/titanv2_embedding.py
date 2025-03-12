from typing import List, Dict

from flotorch_core.chunking.chunking import Chunk
from .embedding_registry import register
from .titanv1_embedding import TitanV1Embedding

"""
This class is responsible for embedding the text using the TitanV2 model.
"""
@register("amazon.titan-embed-text-v2:0")
class TitanV2Embedding(TitanV1Embedding):
    """
    Initializes the TitanV2Embedding class.
    :param model_id: The model id of the TitanV2 model.
    :param region: The region of the TitanV2 model.
    :param dimensions: The dimensions of the embedding.
    :param normalize: Normalize the embedding.
    """

    def __init__(self, model_id: str, region: str = "us-east-1", dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(model_id, region, dimensions, normalize)

    """
    Prepares the chunk for embedding.
    :param chunk: The chunk to be embedded.
    :return: The prepared chunk.
    """

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"inputText": chunk.data, "dimensions": self.dimension, "normalize": self.normalize}
