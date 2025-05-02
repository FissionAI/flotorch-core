from flotorch_core.chunking.hierarical_chunking import HieraricalChunker
from flotorch_core.chunking.fixedsize_chunking import FixedSizeChunker

class ChunkingFactory:
    """
    Factory to create chunking strategies based on configuration.
    """
    @staticmethod
    def create_chunker(chunking_strategy: str, chunk_size: int, chunk_overlap: int, parent_chunk_size: int = None):
        """
        Creates and returns a chunker instance based on the specified chunking strategy.

        Args:
            chunking_strategy (str): The chunking strategy to use. Supported values are 
                "hierarchical" and "fixed".
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap size between consecutive chunks.
            parent_chunk_size (int, optional): The size of the parent chunk. Required 
                for the "hierarchical" chunking strategy. Defaults to None.

        Returns:
            Union[HieraricalChunker, FixedSizeChunker]: An instance of the appropriate 
            chunker class based on the specified strategy.

        Raises:
            ValueError: If an unsupported chunking strategy is provided.
        """
        if chunking_strategy.lower() == "hierarchical":
            return HieraricalChunker(chunk_size, chunk_overlap, parent_chunk_size)
        elif chunking_strategy.lower() == "fixed":
            return FixedSizeChunker(chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported chunking type: {chunking_strategy}")
