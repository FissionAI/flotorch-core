import json
from typing import List, Dict, Any
from abc import abstractmethod
import boto3

from flotorch_core.chunking.chunking import Chunk
from flotorch_core.utils.bedrock_retry_handler import BedRockRetryHander
from .embedding import BaseEmbedding, Embeddings, EmbeddingMetadata


class BedRockEmbedding(BaseEmbedding):
    def __init__(self, model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__(model_id, region, dimensions, normalize)
        self._application_json = "application/json"
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    @BedRockRetryHander()
    def embed(self, chunk: Chunk) -> Embeddings:
        payload = self._prepare_chunk(chunk)
        response = self._invoke_model(payload)
        metadata = self._extract_metadata(response)
        model_response = self._parse_model_response(response)
        return Embeddings(embeddings=self.extract_embedding(model_response),
                          metadata=metadata, text=chunk.data)

    def _invoke_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.invoke_model(
            modelId=self.model_id,
            contentType=self._application_json,
            accept=self._application_json,
            body=json.dumps(payload)
        )

    def _extract_metadata(self, response: Dict[str, Any]) -> EmbeddingMetadata:
        if not response or 'ResponseMetadata' not in response:
            return EmbeddingMetadata(input_tokens=0, latency_ms=0)

        headers = response['ResponseMetadata'].get('HTTPHeaders', {})
        return EmbeddingMetadata(
            input_tokens=headers.get('x-amzn-bedrock-input-token-count', 0),
            latency_ms=headers.get('x-amzn-bedrock-invocation-latency', 0)
        )

    def _parse_model_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if 'body' not in response:
            raise ValueError("Invalid response format: 'body' not found.")
        return json.loads(response['body'].read())

    def extract_embedding(self, response: Dict[str, Any]) -> List[float]:
        return response["embeddings"]
