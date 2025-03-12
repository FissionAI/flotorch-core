import logging
import boto3
from typing import List, Dict, Optional

from flotorch_core.logger.global_logger import get_logger

logger = get_logger()

class BedrockReranker:
    def __init__(self, region: str, rerank_model_id: str, bedrock_client: Optional[boto3.client] = None):
        """
        Initializes the DocumentReranker with AWS region, model ID, and an optional Bedrock client.

        Args:
            region (str): The AWS region to use.
            rerank_model_id (str): The model ID for reranking.
            bedrock_client (boto3.client, optional): Pre-initialized Bedrock agent runtime client.
        """
        self.region = region
        self.rerank_model_id = rerank_model_id
        self.bedrock_agent_runtime = bedrock_client or boto3.client('bedrock-agent-runtime', region_name=region)

    def rerank_documents(self, input_prompt: str, retrieved_documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Reranks a list of documents based on a query using Amazon Bedrock's reranking model.

        Args:
            input_prompt (str): The query for reranking.
            retrieved_documents (List[Dict[str, str]]): List of documents to be reranked.

        Returns:
            List[Dict[str, str]]: A list of reranked documents in order of relevance.
        """
        if not retrieved_documents:
            logger.warning("No documents provided for reranking.")
            return []

        try:
            model_package_arn = f"arn:aws:bedrock:{self.region}::foundation-model/{self.rerank_model_id}"
            rerank_return_count = len(retrieved_documents)

            # Prepare document sources
            document_sources = [
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": doc["text"]}
                    }
                }
                for doc in retrieved_documents
            ]

            # Call the Bedrock API
            response = self.bedrock_agent_runtime.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": input_prompt}}],
                sources=document_sources,
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "numberOfResults": rerank_return_count,
                        "modelConfiguration": {"modelArn": model_package_arn}
                    }
                }
            )

            # Validate response
            results = response.get("results", [])
            if not results:
                logger.error("Reranking failed: No results in response.")
                return []

            # Extract and return reranked documents
            reranked_documents = [
                {"text": retrieved_documents[result["index"]]["text"]}
                for result in results if isinstance(result, dict) and "index" in result
            ]

            logger.info(f"Successfully reranked {len(reranked_documents)} documents.")
            return reranked_documents

        except Exception as e:
            logger.exception(f"Unexpected error during reranking: {e}")

        return []