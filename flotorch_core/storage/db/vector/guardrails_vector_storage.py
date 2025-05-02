from flotorch_core.chunking.chunking import Chunk
from flotorch_core.storage.db.vector.vector_storage import VectorStorage, VectorStorageSearchResponse
from flotorch_core.guardrails.guardrails import BaseGuardRail

class GuardRailsVectorStorage(VectorStorage):
    """
    A wrapper for VectorStorage that applies guardrails to the input and context.
    """

    def __init__(self, vectorStorage: VectorStorage, base_guardrail: BaseGuardRail, 
                 apply_prompt=False, apply_context=False):
        """
        Initializes the GuardRailsVectorStorage.
        Args:
            vectorStorage (VectorStorage): The underlying vector storage.
            base_guardrail (BaseGuardRail): The guardrail to apply.
            apply_prompt (bool): Whether to apply guardrails to the input prompt.
            apply_context (bool): Whether to apply guardrails to the context.
        """
        self.vectorStorage = vectorStorage
        self.base_guardrail = base_guardrail
        self.apply_prompt = apply_prompt
        self.apply_context = apply_context
    
    def search(self, chunk: Chunk, knn: int, hierarchical=False):
        """
        Searches the vector storage for the given chunk.
        Args:
            chunk (Chunk): The chunk to search for.
            knn (int): The number of nearest neighbors to return.
            hierarchical (bool): Whether to use hierarchical search.
        Returns:
            VectorStorageSearchResponse: The search response.
        """
        if self.apply_prompt:
            guardrail_response = self.base_guardrail.apply_guardrail(chunk.data, 'INPUT')
            if guardrail_response['action'] == 'GUARDRAIL_INTERVENED':
                return VectorStorageSearchResponse(
                    status=False,
                    metadata={
                        'guardrail_output': guardrail_response['outputs'][0]['text'],
                        'guardrail_input_assessment': guardrail_response.get('assessments', []),
                        'block_level': 'INPUT',
                        'guardrail_blocked': True
                    }
                )
            
            results = self.vectorStorage.search(chunk, knn, hierarchical)

        if self.apply_context:
            result_text = ' '.join(record.text for record in results.result)
            guardrail_response = self.base_guardrail.apply_guardrail(result_text, 'INPUT')
            if guardrail_response['action'] == 'GUARDRAIL_INTERVENED':
                return VectorStorageSearchResponse(
                    status=False,
                    result=results.result,
                    metadata={
                        'guardrail_output': guardrail_response['outputs'][0]['text'],
                        'guardrail_context_assessment': guardrail_response.get('assessments', []),
                        'block_level': 'CONTEXT',
                        'guardrail_blocked': True,
                        'embedding_metadata': results.metadata['embedding_metadata'] if 'embedding_metadata' in results.metadata else {}
                    }
                )
            
        return results
    
    def embed_query(self, embedding, knn, hierarical=False):
        """
        Embeds the query using the underlying vector storage.
        Args:
            embedding: The embedding to use.
            knn (int): The number of nearest neighbors to return.
            hierarical (bool): Whether to use hierarchical search.
        Returns:
            VectorStorageSearchResponse: The search response.
        """
        self.vectorStorage.embed_query(embedding, knn, hierarical)

    def write(self, body):
        """
        Writes the body to the vector storage.
        Args:
            body: The body to write.
        """
        self.vectorStorage.write(body)

    def read(self, body):
        """
        Reads the body from the vector storage.
        Args:
            body: The body to read.
        """
        self.vectorStorage.read(body)