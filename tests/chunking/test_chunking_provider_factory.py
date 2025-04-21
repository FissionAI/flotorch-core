import pytest
from flotorch_core.chunking.chunking_provider_factory import ChunkingFactory
from flotorch_core.chunking.fixedsize_chunking import FixedSizeChunker
from flotorch_core.chunking.hierarical_chunking import HieraricalChunker

def test_create_hierarchical_chunker():
    """Test creation of hierarchical chunker"""
    factory = ChunkingFactory()
    chunk_size = 100
    chunk_overlap = 20
    parent_chunk_size = 500
    
    chunker = factory.create_chunker(
        chunking_strategy="hierarchical",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_chunk_size=parent_chunk_size
    )
    
    assert isinstance(chunker, HieraricalChunker)
    # Account for token conversion (multiply by tokens_per_charecter)
    assert chunker.chunk_size == chunk_size * chunker.tokens_per_charecter
    assert chunker.chunk_overlap == chunk_overlap * chunker.tokens_per_charecter
    assert chunker.parent_chunk_size == parent_chunk_size * chunker.tokens_per_charecter


def test_create_fixed_size_chunker():
    """Test creation of fixed size chunker"""
    factory = ChunkingFactory()
    chunk_size = 100
    chunk_overlap = 20
    
    chunker = factory.create_chunker(
        chunking_strategy="fixed",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    assert isinstance(chunker, FixedSizeChunker)
    # Account for token conversion (multiply by tokens_per_charecter)
    assert chunker.chunk_size == chunk_size * chunker.tokens_per_charecter
    assert chunker.chunk_overlap == chunk_overlap * chunker.tokens_per_charecter


def test_invalid_chunking_strategy():
    """Test error handling for invalid chunking strategy"""
    factory = ChunkingFactory()
    with pytest.raises(ValueError) as exc_info:
        factory.create_chunker(
            chunking_strategy="invalid",
            chunk_size=100,
            chunk_overlap=20
        )
    assert "Unsupported chunking type: invalid" in str(exc_info.value)

def test_case_insensitive_strategy():
    """Test that chunking strategy is case insensitive"""
    factory = ChunkingFactory()
    chunker1 = factory.create_chunker(
        chunking_strategy="HIERARCHICAL",
        chunk_size=100,
        chunk_overlap=20,
        parent_chunk_size=500
    )
    chunker2 = factory.create_chunker(
        chunking_strategy="Fixed",
        chunk_size=100,
        chunk_overlap=20
    )
    
    assert isinstance(chunker1, HieraricalChunker)
    assert isinstance(chunker2, FixedSizeChunker)

# def test_missing_parent_chunk_size():
#     """Test hierarchical chunker without parent_chunk_size"""
#     factory = ChunkingFactory()
#     chunk_size = 100
#     chunk_overlap = 20
    
#     chunker = factory.create_chunker(
#         chunking_strategy="hierarchical",
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
    
#     assert isinstance(chunker, HieraricalChunker)
#     # Account for token conversion (multiply by tokens_per_charecter)
#     assert chunker.chunk_size == chunk_size * chunker.tokens_per_charecter
#     assert chunker.chunk_overlap == chunk_overlap * chunker.tokens_per_charecter
#     assert chunker.parent_chunk_size is None


def test_negative_chunk_size():
    """Test that negative chunk_size raises ValueError"""
    factory = ChunkingFactory()
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_chunker(
            chunking_strategy="fixed",
            chunk_size= -100,
            chunk_overlap=20
        )
    assert "chunk_size must be positive" in str(exc_info.value)

def test_overlap_greater_than_chunk_size():
    """Test that chunk_overlap > chunk_size raises ValueError"""
    factory = ChunkingFactory()
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_chunker(
            chunking_strategy="fixed",
            chunk_size=100,
            chunk_overlap=150
        )
    assert "chunk_overlap must be less than chunk_size" in str(exc_info.value)

def test_zero_chunk_size():
    """Test that zero chunk_size raises ValueError"""
    factory = ChunkingFactory()
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_chunker(
            chunking_strategy="fixed",
            chunk_size=0,
            chunk_overlap=0
        )
    assert "chunk_size must be positive" in str(exc_info.value)