import os
from opensearchpy import OpenSearch
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.embedding.embedding import BaseEmbedding
from flotorch_core.storage.db.vector.vector_storage import VectorStorage, VectorStorageSearchItem, VectorStorageSearchResponse
from typing import List, Optional

class OpenSearchClient(VectorStorage):
    """
    This class is responsible for storing the data in the OpenSearch.
    """
    def __init__(self, host, port, username, password, index, use_ssl=True, verify_certs=False, ssl_assert_hostname=False, ssl_show_warn=False,
                 embedder: Optional[BaseEmbedding] = None):
        """
        Initializes the OpenSearch client.
        Args:
            host (str): The host of the OpenSearch server.
            port (int): The port of the OpenSearch server.
            username (str): The username to connect to the OpenSearch server.
            password (str): The password to connect to the OpenSearch server.
            index (str): The index name to store the data.
            use_ssl (bool): Whether to use SSL or not. Default is True.
            verify_certs (bool): Whether to verify the SSL certificates or not. Default is False.
            ssl_assert_hostname (bool): Whether to assert the hostname or not. Default is False.
            ssl_show_warn (bool): Whether to show warnings or not. Default is False.
        """
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
        """
        Writes the data to the OpenSearch index.
        Args:
            body (dict): The data to be written to the index.
        Returns:
            dict: The response from the OpenSearch server.
        """
        return self.client.index(index=self.index, body=body)
    
    def read(self, body):
        """
        Reads the data from the OpenSearch index.
        Args:
            body (dict): The data to be read from the index.
        Returns:
            dict: The response from the OpenSearch server.
        """
        return self.search(self.index, body)
    
    def write_bulk(self, body: List[dict]):
        return self.client.bulk(body=body)

    # TODO: Need to create a model class for the return type of the search method
    # This model class has to be created in the base class and this return type has to be consitent in all the vector_sotrage classes
    def search(self, chunk: Chunk,  knn: int, hierarchical=False):
        """
        Searches the data in the OpenSearch index.
        Args:
            chunk (Chunk): The chunk to be searched.
            knn (int): The number of nearest neighbors to be returned.
            hierarchical (bool): Whether to use hierarchical search or not. Default is False.
        Returns:
            VectorStorageSearchResponse: The response from the OpenSearch server.
        """
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
        """
        Embeds the query vector for the OpenSearch index.
        Args:
            query_vector (List[float]): The query vector to be embedded.
            knn (int): The number of nearest neighbors to be returned.
            hierarchical (bool): Whether to use hierarchical search or not. Default is False.
        Returns:
            dict: The query to be sent to the OpenSearch server.
        """
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