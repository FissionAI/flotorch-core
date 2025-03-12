from typing import List

from langchain.text_splitter import CharacterTextSplitter

from .chunking import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """
    This class is responsible for chunking the text into fixed size chunks.
    :param chunk_size: The size of the chunk.
    :param chunk_overlap: The overlap between chunks.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__()
        self.chunk_size = self.tokens_per_charecter * chunk_size
        self.chunk_overlap = int(chunk_overlap * self.chunk_size / 100)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    """
    Chunks the text into fixed size chunks.
    :param data: The input text.
    :return: The list of chunks."""
    def chunk(self, data: str) -> List[Chunk]:
        if not data:
            raise ValueError("Input text cannot be empty or None")

        data = self._clean_data(data)
        self.text_splitter = CharacterTextSplitter(
            separator=self.space,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        chunks = self.text_splitter.split_text(data)
        return [Chunk(chunk) for chunk in chunks]
