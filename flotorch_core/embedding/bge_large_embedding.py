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
        super().__init__(model_id, region, config.get_sagemaker_arn_role(), dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"text_inputs": [chunk.data], "mode": "embedding"}


@register("huggingface-sentencesimilarity-bge-m3")
class BGEM3Embedding(SageMakerEmbedder):
    """
    BGE M3 Hugging Face model for sentence similarity.
    """
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(model_id, region, config.get_sagemaker_arn_role(), dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"text_inputs": [chunk.data], "mode": "embedding"}


@register("huggingface-textembedding-gte-qwen2-7b-instruct")
class GTEQwen2Embedding(SageMakerEmbedder):
    """
    GTE Qwen2-7B Instruct Hugging Face model for text embedding.
    """
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(model_id, region, config.get_sagemaker_arn_role(), dimensions, normalize)

    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        return {"inputs": [chunk.data]}
