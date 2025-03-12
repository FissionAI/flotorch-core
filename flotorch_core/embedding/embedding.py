from abc import ABC, abstractmethod
import re
from typing import List, Dict

from flotorch_core.chunking.chunking import Chunk

"""
This class is responsible for embedding the text using the Llama model.
"""
class EmbeddingMetadata:
    """
    Initializes the EmbeddingMetadata class.
    :param input_tokens: The number of input tokens.
    :param latency_ms: The latency in milliseconds.
    """
    def __init__(self, input_tokens: int, latency_ms: int):
        self.input_tokens = input_tokens
        self.latency_ms = latency_ms
    
    def append(self, metadata: 'EmbeddingMetadata'):
        self.input_tokens += int(metadata.input_tokens)
        self.latency_ms += int(metadata.latency_ms)

    def to_json(self):
        return {
            'input_token': self.input_tokens,
            'latency_ms': self.latency_ms
        }


class Embeddings:
    """
    Initializes the Embeddings class.
    :param embeddings: The embeddings.
    :param metadata: The metadata.
    """
    def __init__(self, embeddings: List[List[float]], metadata: EmbeddingMetadata, text: str):
        self.embeddings = embeddings
        self.metadata = metadata
        self.text = text
        self.id = ''

    def clean_text_for_vector_db(self, text):
        """
        Cleans the input text by removing quotes, special symbols, extra whitespaces,
        newline (\n), and tab (\t) characters.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text.
        """
        # Remove single and double quotes
        text = text.replace('"', '').replace("'", "")
        # Remove special symbols (keeping alphanumerics and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove newlines and tabs
        text = text.replace('\n', ' ').replace('\t', ' ')
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading and trailing spaces
        return text.strip()

    def to_json(self) -> Dict:
        return {
            "vectors": self.embeddings,
            "text": self.clean_text_for_vector_db(self.text),
            "metadata": {
                    "inputTokens": self.metadata.input_tokens,
                    "latencyMs": self.metadata.latency_ms
                }
        }

class EmbeddingList:
    def __init__(self):
        self.embeddings: List[Embeddings] = []
        self.metadata = EmbeddingMetadata(0, 0)

    def append(self, embeddings: Embeddings):
        self.embeddings.append(embeddings)
        self.metadata.append(embeddings.metadata)

"""
This class is responsible for embedding the text."""
class BaseEmbedding(ABC):
    """
    Initializes the BaseEmbedding class.
    :param dimensions: The dimensions of the embedding.
    :param normalize: Normalize the embedding.
    """

    def __init__(self,  model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__()
        self.model_id = model_id
        self.region = region
        self.dimension = dimensions
        self.normalize = normalize

    """
    Prepares the chunk for embedding.
    :param chunk: The chunk to be embedded.
    :return: The prepared chunk.
    """
    @abstractmethod
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        pass

    """
    Embeds the chunk.
    :param chunk: The chunk to be embedded.
    :return: The embeddings.
    """
    @abstractmethod
    def embed(self, chunk: Chunk) -> Embeddings:
        pass

    """
    Embeds the list of chunks.
    :param chunks: The list of chunks to be embedded.
    :return: The list of embeddings.
    """
    def embed_list(self, chunks: List[Chunk]) -> EmbeddingList:
        embedding_list = EmbeddingList()
        if not isinstance(chunks, list):
            return embedding_list.append(self.embed(chunks))
        for chunk in chunks:
            if chunk.child_data:
                for child_chunk in chunk.child_data:
                    embedding = self.embed(child_chunk)
                    embedding.id = chunk.id
                    embedding.text = chunk.data
                    embedding_list.append(embedding)
            else:
                embedding = self.embed(chunk)
                embedding.id = chunk.id
                embedding_list.append(embedding)
        return embedding_list