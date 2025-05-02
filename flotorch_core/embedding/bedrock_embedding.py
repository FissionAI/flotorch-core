import json
from typing import List, Dict, Any
from abc import abstractmethod
import boto3

from flotorch_core.chunking.chunking import Chunk
from flotorch_core.utils.bedrock_retry_handler import BedRockRetryHander
from .embedding import BaseEmbedding, Embeddings, EmbeddingMetadata


class BedRockEmbedding(BaseEmbedding):
    """
    Base class for Bedrock Embedding models."""
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        """
        Initialize the BedRockEmbedding class.
        Args:
            model_id (str): The model ID for the Bedrock model.
            region (str): The AWS region where the model is hosted.
            dimensions (int): The number of dimensions for the embedding. Default is 256.
            normalize (bool): Whether to normalize the embedding. Default is True.
        Returns:
            None
        """
        super().__init__(model_id, region, dimensions, normalize)
        self._application_json = "application/json"
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    @BedRockRetryHander()
    def embed(self, chunk: Chunk) -> Embeddings:
        """
        Generate embeddings for a given chunk of text.
        Args:
            chunk (Chunk): The chunk of text to generate embeddings for.
        Returns:
            Embeddings: An object containing the generated embeddings and metadata.
        """
        payload = self._prepare_chunk(chunk)
        response = self._invoke_model(payload)
        metadata = self._extract_metadata(response)
        model_response = self._parse_model_response(response)
        return Embeddings(embeddings=self.extract_embedding(model_response),
                          metadata=metadata, text=chunk.data)

    def _invoke_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the Bedrock model with the given payload.
        Args:
            payload (Dict[str, Any]): The payload to send to the model.
        Returns:
            Dict[str, Any]: The response from the model.
        """
        return self.client.invoke_model(
            modelId=self.model_id,
            contentType=self._application_json,
            accept=self._application_json,
            body=json.dumps(payload)
        )

    def _extract_metadata(self, response: Dict[str, Any]) -> EmbeddingMetadata:
        """
        Extract metadata from the model response.
        Args:
            response (Dict[str, Any]): The response from the model.
        Returns:
            EmbeddingMetadata: An object containing the metadata.
        """
        if not response or 'ResponseMetadata' not in response:
            return EmbeddingMetadata(input_tokens=0, latency_ms=0)

        headers = response['ResponseMetadata'].get('HTTPHeaders', {})
        return EmbeddingMetadata(
            input_tokens=headers.get('x-amzn-bedrock-input-token-count', 0),
            latency_ms=headers.get('x-amzn-bedrock-invocation-latency', 0)
        )

    def _parse_model_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the model response to extract the body.
        Args:
            response (Dict[str, Any]): The response from the model.
        Returns:
            Dict[str, Any]: The parsed response body.
        """
        if 'body' not in response:
            raise ValueError("Invalid response format: 'body' not found.")
        return json.loads(response['body'].read())

    def extract_embedding(self, response: Dict[str, Any]) -> List[float]:
        """
        Extract the embedding from the model response.
        Args:
            response (Dict[str, Any]): The model response.
        Returns:
            List[float]: The extracted embedding.
        """
        return response["embeddings"]
