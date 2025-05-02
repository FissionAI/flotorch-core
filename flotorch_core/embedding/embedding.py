from abc import ABC, abstractmethod
import re
from typing import List, Dict

from flotorch_core.chunking.chunking import Chunk


class EmbeddingMetadata:
    """
    This class is responsible for embedding the text using the Llama model.
    """
   
    def __init__(self, input_tokens: int, latency_ms: int):
        """
        Initializes the EmbeddingMetadata class.
        Args:
            input_tokens (int): The number of input tokens.
            latency_ms (int): The latency in milliseconds.
        Returns:
            None
        """
        self.input_tokens = input_tokens
        self.latency_ms = latency_ms
    
    def append(self, metadata: 'EmbeddingMetadata'):
        """
        Appends the metadata of another EmbeddingMetadata object to this one.
        Args:
            metadata (EmbeddingMetadata): The metadata to append.
        Returns:
            None
        """
        self.input_tokens += int(metadata.input_tokens)
        self.latency_ms += int(metadata.latency_ms)

    def to_json(self):
        """
        Converts the metadata to JSON format.
        Returns:
            dict: The metadata in JSON format.
        """
        return {
            'input_token': self.input_tokens,
            'latency_ms': self.latency_ms
        }


class Embeddings:
    """
    This class is responsible for embedding the text"""
    
    def __init__(self, embeddings: List[List[float]], metadata: EmbeddingMetadata, text: str):
        """
        Initializes the Embeddings class.
        Args:
            embeddings (List[List[float]]): The embeddings of the text.
            metadata (EmbeddingMetadata): The metadata of the embeddings.
            text (str): The original text.
        Returns:
            None
        """
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
        """
        Converts the embeddings to JSON format.
        Returns:
            dict: The embeddings in JSON format.
        """
        return {
            "vectors": self.embeddings,
            "text": self.clean_text_for_vector_db(self.text),
            "metadata": {
                    "inputTokens": self.metadata.input_tokens,
                    "latencyMs": self.metadata.latency_ms
                }
        }

class EmbeddingList:
    """
    This class is responsible for creating a list of embeddings."""

    def __init__(self):
        """
        Initializes the EmbeddingList class.
        """
        self.embeddings: List[Embeddings] = []
        self.metadata = EmbeddingMetadata(0, 0)

    def append(self, embeddings: Embeddings):
        """
        Appends the embeddings to the list.
        Args:
            embeddings (Embeddings): The embeddings to append.
        Returns:
            None
        """
        self.embeddings.append(embeddings)
        self.metadata.append(embeddings.metadata)


class BaseEmbedding(ABC):
    """
    This is the base class for all embedding models
    """

    def __init__(self,  model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        """
        Initializes the BaseEmbedding class.
        Args:
            model_id (str): The ID of the model.
            region (str): The region of the model.
            dimensions (int): The dimensions of the embeddings.
            normalize (bool): Whether to normalize the embeddings.
        Returns:
            None
        """
        super().__init__()
        self.model_id = model_id
        self.region = region
        self.dimension = dimensions
        self.normalize = normalize
    
    @abstractmethod
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        """
        Prepares the chunk for embedding.
        Args:
            chunk (Chunk): The chunk to be prepared.
        Returns:
            The prepared chunk.
        """
        pass

    
    @abstractmethod
    def embed(self, chunk: Chunk) -> Embeddings:
        """
        Embeds the chunk.
        Args:
            chunk (Chunk): The chunk to be embedded.
        Returns:
            The prepared chunk.
        """
        pass

    
    def embed_list(self, chunks: List[Chunk]) -> EmbeddingList:
        """
        Embeds the list of chunks.
        Args:
            chunks (List[Chunk]): The list of chunks to be embedded.
        Returns:
            EmbeddingList: The list of embeddings.
        """
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