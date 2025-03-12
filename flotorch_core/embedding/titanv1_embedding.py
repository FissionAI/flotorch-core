from typing import List, Dict

from flotorch_core.chunking.chunking import Chunk
from .bedrock_embedding import BedRockEmbedding
from .embedding_registry import register


@register("amazon.titan-embed-image-v1")
@register("amazon.titan-text-express-v1")
class TitanV1Embedding(BedRockEmbedding):
    """
    Initializes the TitanV1Embedding class.
    :param model_id: The model id of the TitanV1 model.
    :param region: The region of the TitanV1 model.
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
        return {"inputText": chunk.data, "embeddingConfig": {"outputEmbeddingLength": self.dimension}}

    """
    Extracts the embedding from the response.
    :param response: The response from the model.
    :return: The embedding.
    """
    def extract_embedding(self, response: Dict) -> List[float]:
        return response["embedding"]
