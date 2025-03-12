from flotorch_core.storage.db.vector.vector_storage import VectorStorage
from flotorch_core.rerank.rerank import BedrockReranker

class RerankedVectorStorage(VectorStorage):

    def __init__(self, vectorStorage: VectorStorage, bedrockReranker: BedrockReranker):
        self.vectorStorage = vectorStorage
        self.reranker = bedrockReranker
    
    def search(self, query: str, knn: int, hierarchical=False):
        results = self.vectorStorage.search(query, knn, hierarchical)
        return self.reranker.rerank_documents(query, results)

    def embed_query(self, embedding, knn, hierarical=False):
        self.vectorStorage.embed_query(embedding, knn, hierarical)