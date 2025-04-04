import pytest
from flotorch_core.embedding.titanv1_embedding import TitanV1Embedding
from flotorch_core.logger.global_logger import get_logger

logger = get_logger()

class TestTitanV1Embedding:
    """Test suite for TitanV1Embedding class"""
    
    def test_valid_initialization(self):
        """Test successful initialization with valid parameters"""
        logger.info("Testing valid initialization")
        embedder = TitanV1Embedding(
            model_id="amazon.titan-embed-text-v1",
            region="us-east-1",
            dimensions=256,
            normalize=True
        )
        assert embedder.model_id == "amazon.titan-embed-text-v1"
        assert embedder.region == "us-east-1"
        assert embedder.dimension == 256
        assert embedder.normalize is True

    def test_default_parameters(self):
        """Test initialization with default parameters"""
        logger.info("Testing default parameters")
        embedder = TitanV1Embedding(model_id="amazon.titan-embed-text-v1",  region="us-east-1")
        assert embedder.region == "us-east-1"
        assert embedder.dimension == 256
        assert embedder.normalize is True

    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        logger.info("Testing custom parameters")
        embedder = TitanV1Embedding(
            model_id="amazon.titan-embed-text-v1",
            region="eu-west-1",
            dimensions=512,
            normalize=False
        )
        assert embedder.model_id == "amazon.titan-embed-text-v1"
        assert embedder.region == "eu-west-1"
        assert embedder.dimension == 512
        assert embedder.normalize is False
