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
    Initializes the CohereEmbedding class.
    :param model_id: The model id of the Cohere model.
    :param region: The region of the Cohere model.
    :param dimensions: The dimensions of the embedding.
    :param normalize: Normalize the embedding.
    """

    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(model_id, region, dimensions, normalize)

    """
    Prepares the chunk for embedding.
    :param chunk: The chunk to be embedded.
    :return: The prepared chunk.
    """
    
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"texts": [chunk.data], "input_type": "search_document"}
    
    def extract_embedding(self, response: Dict[str, Any]) -> List[float]:
        return response["embeddings"][0]

