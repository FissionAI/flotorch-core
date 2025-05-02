from typing import List, Dict

from flotorch_core.chunking.chunking import Chunk
from .bedrock_embedding import BedRockEmbedding
from .embedding_registry import register


@register("amazon.titan-embed-image-v1")
@register("amazon.titan-text-express-v1")
class TitanV1Embedding(BedRockEmbedding):
    """
    This class is used to create embeddings using the TitanV1 model.
    It inherits from the BedRockEmbedding class.
    """

    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        """
        Initializes the TitanV1Embedding class.
        Args:
            model_id (str): The model ID.
            region (str): The region.
            dimensions (int): The dimensions of the embedding. Default is 256.
            normalize (bool): Whether to normalize the embedding. Default is True.
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
        return {"inputText": chunk.data, "embeddingConfig": {"outputEmbeddingLength": self.dimension}}

    
    def extract_embedding(self, response: Dict) -> List[float]:
        """
        Extracts the embedding from the response.
        Args:
            response (Dict): The response from the model.
        Returns:
            List[float]: The extracted embedding.
        """
        return response["embedding"]
