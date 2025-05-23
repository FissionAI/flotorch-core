from abc import ABC, abstractmethod
from typing import List, Dict


class BaseReranker(ABC):
    """
    Abstract base class for document rerankers.
    Defines the interface for reranking documents.
    """

    def __init__(self, region: str, rerank_model_id: str):
        """
        Initializes the BaseReranker with AWS region and model ID.

        Args:
            region (str): The AWS region to use.
            rerank_model_id (str): The model ID for reranking.
        """
        self.region = region
        self.rerank_model_id = rerank_model_id

    @abstractmethod
    def rerank_documents(self, input_prompt: str, retrieved_documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Reranks a list of documents based on a query.

        Args:
            input_prompt (str): The query for reranking.
            retrieved_documents (List[Dict[str, str]]): List of documents to be reranked.
                                                        Each document is expected to be a dictionary
                                                        with at least a 'text' key.

        Returns:
            List[Dict[str, str]]: A list of reranked documents in order of relevance.
                                  The structure of the returned documents should match the
                                  input documents (e.g., still include 'text' key).
        """
        pass