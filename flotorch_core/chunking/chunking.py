import uuid
from abc import ABC, abstractmethod
from typing import List


class Chunk:
    """
    A class to represent a chunk of text.
    """

    def __init__(self, data):
        """
        Constructs a new Chunk object.
        Args:
            data (str): The data of the chunk.
        """
        self.id = str(uuid.uuid4())
        self.data = data
        self.child_data = None

    def add_child(self, child_data):
        """
        Adds a child to the chunk.
        Args:
            child_data (str): The data of the child.
        """
        if not self.child_data:
            self.child_data = []
        self.child_data.append(child_data)

    def __str__(self):
        """
        Returns a string representation of the Chunk object.
        """
        return f"Parent Chunk: {self.data}, Chunk: {self.child_data}"


class BaseChunker(ABC):
    """
    Abstract base class for chunking text.
    """

    def __init__(self):
        """
        Constructs a new BaseChunker object.
        """
        super().__init__()
        self.space = ' '
        self.separators = [' ', '\t', '\n', '\r', '\f', '\v']
        self.tokens_per_charecter = 4

    @abstractmethod
    def chunk(self, data: str) -> List[Chunk]:
        """
        Chunks the input text.
        Args:
            data (str): The input text to chunk.
        Returns:
            List[Chunk]: A list of Chunk objects.  
        """
        pass

    def chunk_list(self, data: List[str]) -> List[Chunk]:
        """
        Chunks a list of input texts.
        Args:
            data (List[str]): The list of input texts to chunk.
        Returns:
            List[Chunk]: A list of Chunk objects.  
        """
        result = []
        for d in data:
            result.extend(self.chunk(d))
        return result

    def _clean_data(self, data: str) -> str:
        """
        Cleans the input text.
        Args:
            data (str): The input text to clean.
        Returns:
            str: The cleaned text.
        """
        for sep in self.separators:
            data = data.replace(sep, self.space)
        return data
