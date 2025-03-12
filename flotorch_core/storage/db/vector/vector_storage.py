from dataclasses import dataclass, field
import json
from flotorch_core.storage.db.db_storage import DBStorage
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from flotorch_core.embedding.embedding import BaseEmbedding

@dataclass
class VectorStorageSearchItem:
    text: str
    execution_id: Optional[str] = None
    chunk_id: Optional[str] = None
    parent_id: Optional[str] = None
    vectors: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self):
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
    status: bool
    result: List[VectorStorageSearchItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self):
        return {
            "status": self.status,
            "result": [item.to_json() for item in self.result],
            "metadata": self.metadata
        }

    
class VectorStorage(DBStorage, ABC):
    def __init__(self, embedder: Optional[BaseEmbedding] = None):
        self.embedder = embedder
        
    
    @abstractmethod
    def search(self, query: str, knn, hierarchical=False):
        pass

    @abstractmethod
    def embed_query(self, embedding, knn, hierarical=False):
        pass
    
