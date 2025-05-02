from typing import List, Dict

from flotorch_core.chunking.chunking import Chunk
from .embedding_registry import register
from .titanv1_embedding import TitanV1Embedding


@register("amazon.titan-embed-text-v2:0")
class TitanV2Embedding(TitanV1Embedding):
    """
    This class is responsible for embedding the text using the TitanV2 model.
    """

    def __init__(self, model_id: str, region: str = "us-east-1", dimensions: int = 256, normalize: bool = True) -> None:
        """
        Initializes the TitanV2Embedding class.
        Args:
            model_id (str): The model ID.
            region (str): The region where the model is hosted.
            dimensions (int): The number of dimensions for the embedding.
            normalize (bool): Whether to normalize the embedding or not.
        Returns:
            None
        """
        super().__init__(model_id, region, dimensions, normalize)

    

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        """
        Prepares the chunk for embedding.
        Args:
            chunk (Chunk): The chunk to be prepared.
        Returns:
            Dict: The prepared chunk.
        """
        return {"inputText": chunk.data, "dimensions": self.dimension, "normalize": self.normalize}
