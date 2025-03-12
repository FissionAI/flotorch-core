import os
from opensearchpy import OpenSearch
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.embedding.embedding import BaseEmbedding
from flotorch_core.storage.db.vector.vector_storage import VectorStorage, VectorStorageSearchItem, VectorStorageSearchResponse
from typing import List, Optional

"""
This class is responsible for storing the data in the OpenSearch.
"""

class OpenSearchClient(VectorStorage):
    def __init__(self, host, port, username, password, index, use_ssl=True, verify_certs=False, ssl_assert_hostname=False, ssl_show_warn=False,
                 embedder: Optional[BaseEmbedding] = None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.index = index
        self.embedder = embedder
        
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=(self.username, self.password),
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_assert_hostname=ssl_assert_hostname,
            ssl_show_warn=ssl_show_warn,
        )
    
    def write(self, body):
        return self.client.index(index=self.index, body=body)
    
    def read(self, body):
        return self.search(self.index, body)
    
    def write_bulk(self, body: List[dict]):
        return self.client.bulk(body=body)

    # TODO: Need to create a model class for the return type of the search method
    # This model class has to be created in the base class and this return type has to be consitent in all the vector_sotrage classes
    def search(self, chunk: Chunk,  knn: int, hierarchical=False):
        embedding = self.embedder.embed(chunk)
        query_vector = embedding.embeddings
        body = self.embed_query(query_vector, knn, hierarchical)
        response = self.client.search(index=self.index, body=body)

        result = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            result.append(
                VectorStorageSearchItem(
                    execution_id=hit['_id'],
                    chunk_id=source['chunk_id'] if 'chunk_id' in source else None,
                    parent_id=source['parent_id'] if 'parent_id' in source else None,
                    text=source['text'],
                    vectors=source['vectors'],
                    metadata=source['metadata']
                )
            )

        return VectorStorageSearchResponse(
            status=True,
            result=result,
            metadata={
                "embedding_metadata": embedding.metadata
            }
        )
    
    def embed_query(self, query_vector: List[float], knn: int, hierarchical=False):
        vector_field = next((field for field, props in 
                            self.client.indices.get_mapping(index=self.index)[self.index]['mappings']['properties'].items() 
                            if 'type' in props and props['type'] == 'knn_vector'), None)
        query =  {
            "size": knn,
            "query": {
                "knn": {
                    vector_field: {
                        "vector": query_vector,
                        "k": knn
                    }
                }
            },
            "_source": True,
            "fields": ["text", "parent_id"]
        }
        if hierarchical:
            query["collapse"] = {"field": "parent_id.keyword"}

        return query