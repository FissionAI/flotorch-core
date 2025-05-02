from typing import List

from langchain.text_splitter import CharacterTextSplitter

from flotorch_core.chunking.chunking import Chunk
from flotorch_core.chunking.fixedsize_chunking import FixedSizeChunker


class HieraricalChunker(FixedSizeChunker):
    """
    This class is used to chunk text into smaller pieces using a hierarchical approach.
    It first splits the text into larger chunks and then further splits those chunks into smaller ones.
    The class inherits from the FixedSizeChunker class and uses the CharacterTextSplitter from langchain.
    The chunking process is done in two steps:
    1. The text is split into larger chunks using the parent chunk size.
    2. Each of those larger chunks is then split into smaller chunks using the child chunk size.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, parent_chunk_size: int):
        """
        Initialize the HieraricalChunker with chunk size, overlap, and parent chunk size.
        Args:
            chunk_size (int): The size of the child chunks.
            chunk_overlap (int): The overlap between child chunks.
            parent_chunk_size (int): The size of the parent chunks.
        Returns:
            None
        """
        super().__init__(chunk_size, chunk_overlap)
        self.parent_chunk_size = self.tokens_per_charecter * parent_chunk_size
        if self.parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be positive")
        if self.chunk_size > self.parent_chunk_size:
            raise ValueError("child_chunk_size must be less than parent chunking size")

    def chunk(self, data: str) -> List[Chunk]:
        """
        Chunk the input text into smaller pieces using a hierarchical approach.
        Args:
            data (str): The input text to be chunked.
        Returns:
            List[Chunk]: A list of Chunk objects representing the chunked text.
        """
        if not data:
            raise ValueError("Input text cannot be empty or None")

        data = self._clean_data(data)
        self.parent_text_splitter = CharacterTextSplitter(
            separator=self.space,
            chunk_size=self.parent_chunk_size,
            chunk_overlap=0,  # Can change this at a later point of time
            length_function=len,
            is_separator_regex=False
        )
        self.child_text_splitter = CharacterTextSplitter(
            separator=self.space,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        parent_chunks = self.parent_text_splitter.split_text(data)
        overall_chunks = []
        for parent_chunk in parent_chunks:
            chunk_object = Chunk(parent_chunk)
            child_chunks = self.child_text_splitter.split_text(parent_chunk)
            for child_chunk in child_chunks:
                child_chunk_object = Chunk(child_chunk)
                chunk_object.add_child(child_chunk_object)

            overall_chunks.append(chunk_object)
        return overall_chunks
