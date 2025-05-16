from dataclasses import dataclass, field
import json
from flotorch_core.storage.db.db_storage import DBStorage
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from flotorch_core.embedding.embedding import BaseEmbedding

@dataclass
class VectorStorageSearchItem:
    """
    Represents a single item in the vector storage search response.
    """
    text: str
    execution_id: Optional[str] = None
    chunk_id: Optional[str] = None
    parent_id: Optional[str] = None
    vectors: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self):
        """
        Converts the VectorStorageSearchItem to a JSON-serializable dictionary.
        returns:
            dict: A dictionary representation of the VectorStorageSearchItem.
        """
        return {
            "text": self.text,
            "execution_id": self.execution_id,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "vectors": self.vectors,
            "metadata": self.metadata
        }
@dataclass
class VectorStorageSearchResponse:
    """
    Represents the response from a vector storage search.
    """
    status: bool
    result: List[VectorStorageSearchItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self):
        """
        Converts the VectorStorageSearchResponse to a JSON-serializable dictionary.
        returns:
            dict: A dictionary representation of the VectorStorageSearchResponse.
        """
        return {
            "status": self.status,
            "result": [item.to_json() for item in self.result],
            "metadata": self.metadata
        }

    
class VectorStorage(DBStorage, ABC):
    """
    Abstract base class for vector storage systems.
    This class defines the interface for vector storage systems, including methods for
    searching and embedding queries.
    """
    def __init__(self, embedder: Optional[BaseEmbedding] = None):
        """
        Initializes the VectorStorage with an optional embedder.
        Args:
            embedder (BaseEmbedding, optional): An instance of a BaseEmbedding class for embedding queries.
        """
        self.embedder = embedder
        
    
    @abstractmethod
    def search(self, query: str, knn, hierarchical=False):
        """
        Searches the vector storage for the given query.
        Args:
            query (str): The query string to search for.
            knn (int): The number of nearest neighbors to return.
            hierarchical (bool): Whether to use hierarchical search or not."""
        pass

    @abstractmethod
    def embed_query(self, embedding, knn, hierarical=False):
        """
        Embeds the query using the provided embedding.
        Args:
            embedding (BaseEmbedding): The embedding to use for the query.
            knn (int): The number of nearest neighbors to return.
            hierarchical (bool): Whether to use hierarchical search or not.
        """
        pass
    
