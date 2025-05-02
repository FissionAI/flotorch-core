from typing import Any, Dict, List

from flotorch_core.chunking.chunking import Chunk
from .bedrock_embedding import BedRockEmbedding
from .embedding_registry import register

"""
This class is responsible for embedding the text using the Cohere model.
"""
@register("cohere.embed-multilingual-v3")
@register("cohere.embed-english-v3")
class CohereEmbedding(BedRockEmbedding):
    """
    CohereEmbedding class for embedding text using Cohere models.
    """

    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        """
        Initializes the CohereEmbedding class.
        Args:
            model_id (str): The model ID for the Cohere model.
            region (str): The region for the Cohere model.
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
            chunk (Chunk): The chunk to be embedded.
        Returns:
            Dict: A dictionary containing the chunk data and input type.
        """
        return {"texts": [chunk.data], "input_type": "search_document"}
    
    def extract_embedding(self, response: Dict[str, Any]) -> List[float]:
        """
        Extracts the embedding from the response.
        Args:
            response (Dict[str, Any]): The response from the embedding model.
        Returns:
            List[float]: The extracted embedding.
        """
        return response["embeddings"][0]

