from typing import List, Dict
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.config.config import Config
from flotorch_core.config.env_config_provider import EnvConfigProvider
from flotorch_core.embedding.sagemaker_embedding import SageMakerEmbedder
from flotorch_core.embedding.embedding_registry import register

env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)



@register("huggingface-sentencesimilarity-bge-large-en-v1-5")
class BGELargeEmbedding(SageMakerEmbedder):
    """
    BGE Large Hugging Face model for sentence similarity.
    """
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        """
        Initialize the BGE Large embedding model.
        Args:
            model_id (str): The model ID for the Hugging Face model.
            region (str): The AWS region for the SageMaker endpoint.
            dimensions (int): The number of dimensions for the embedding. Default is 256.
            normalize (bool): Whether to normalize the embedding. Default is True.
        Returns:
            None
        """
        super().__init__(model_id, region, config.get_sagemaker_arn_role(), dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        """
        Prepare the chunk for embedding.
        Args:
            chunk (Chunk): The chunk to be embedded.
        Returns:
            Dict: The prepared chunk for embedding.
        """
        return {"text_inputs": [chunk.data], "mode": "embedding"}


@register("huggingface-sentencesimilarity-bge-m3")
class BGEM3Embedding(SageMakerEmbedder):
    """
    BGE M3 Hugging Face model for sentence similarity.
    """
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        """
        Initialize the BGE M3 embedding model.
        Args:
            model_id (str): The model ID for the Hugging Face model.
            region (str): The AWS region for the SageMaker endpoint.
            dimensions (int): The number of dimensions for the embedding. Default is 256.
            normalize (bool): Whether to normalize the embedding. Default is True.
        Returns:
            None
        """
        super().__init__(model_id, region, config.get_sagemaker_arn_role(), dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        """
        Prepare the chunk for embedding.
        Args:
            chunk (Chunk): The chunk to be embedded.
        Returns:
            Dict: The prepared chunk for embedding.
        """
        return {"text_inputs": [chunk.data], "mode": "embedding"}


@register("huggingface-textembedding-gte-qwen2-7b-instruct")
class GTEQwen2Embedding(SageMakerEmbedder):
    """
    GTE Qwen2-7B Instruct Hugging Face model for text embedding.
    """
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        """
        Initialize the GTE Qwen2-7B Instruct embedding model.
        Args:
            model_id (str): The model ID for the Hugging Face model.
            region (str): The AWS region for the SageMaker endpoint.
            dimensions (int): The number of dimensions for the embedding. Default is 256.
            normalize (bool): Whether to normalize the embedding. Default is True.
        Returns:
            None
        """
        super().__init__(model_id, region, config.get_sagemaker_arn_role(), dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        """
        Prepare the chunk for embedding.
        Args:
            chunk (Chunk): The chunk to be embedded.
        Returns:
            Dict: The prepared chunk for embedding.
        """
        return {"inputs": [chunk.data]}
