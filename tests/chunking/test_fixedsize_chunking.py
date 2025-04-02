import pytest
from flotorch_core.chunking.fixedsize_chunking import FixedSizeChunker
from flotorch_core.chunking.chunking import Chunk

@pytest.fixture
def chunker():
    return FixedSizeChunker(chunk_size=100, chunk_overlap=20)

def test_fixed_size_chunker_initialization():
    # Test normal initialization
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_size == 400  # 100 * tokens_per_charecter (4)
    assert chunker.chunk_overlap == 80  # 20% of chunk_size

def test_invalid_chunk_size():
    # Test initialization with invalid chunk size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=0, chunk_overlap=20)
    
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=-1, chunk_overlap=20)

def test_invalid_chunk_overlap():
    # Test initialization with invalid chunk overlap
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)
    
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        FixedSizeChunker(chunk_size=100, chunk_overlap=150)

def test_chunk_empty_input(chunker):
    # Test chunking empty input
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        chunker.chunk("")

def test_chunk_none_input(chunker):
    # Test chunking None input
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        chunker.chunk(None)

def test_basic_chunking(chunker):
    # Test basic chunking functionality
    text = "This is a test text that should be split into multiple chunks based on the specified size."
    chunks = chunker.chunk(text)
    
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert len(chunks) > 0

def test_chunk_overlap(chunker):
    # Test that chunks properly overlap
    text = "a" * 1000  # Create a long text
    chunks = chunker.chunk(text)
    
    # Check if consecutive chunks have overlapping content
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].data
        next_chunk = chunks[i + 1].data
        
        # The end of the current chunk should appear at the start of the next chunk
        assert current_chunk[-80:] == next_chunk[:80]

def test_chunk_size_consistency(chunker):
    # Test that chunks are consistently sized (except possibly the last one)
    text = "This is a test text " * 50  # Create a long text
    chunks = chunker.chunk(text)
    
    # All chunks except the last one should have the same size
    for i in range(len(chunks) - 1):
        assert len(chunks[i].data) <= chunker.chunk_size

def test_chunking_with_different_sizes():
    # Test chunking with different size configurations
    small_chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    large_chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=30)
    
    text = "This is a test text " * 20
    
    small_chunks = small_chunker.chunk(text)
    large_chunks = large_chunker.chunk(text)
    
    # Smaller chunk size should result in more chunks
    assert len(small_chunks) > len(large_chunks)

def test_chunking_with_special_characters(chunker):
    # Test chunking text with special characters
    text = "Special characters: !@#$%^&*()_+ \n\t" * 10
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)

def test_chunking_with_unicode(chunker):
    # Test chunking text with unicode characters
    text = "Unicode characters: 你好世界 αβγδ 👋🌍" * 10
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)

def test_chunking_single_word():
    # Test chunking a single word that's smaller than chunk size
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    text = "Hello"
    chunks = chunker.chunk(text)
    
    assert len(chunks) == 1
    assert chunks[0].data == "Hello"

def test_chunking_exact_size():
    # Test chunking text that exactly matches chunk size
    chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
    text = "12345"  # Exactly 5 characters
    chunks = chunker.chunk(text)
    
    assert len(chunks) == 1
    assert len(chunks[0].data) == 5

def test_chunking_with_spaces(chunker):
    # Test chunking text with multiple spaces
    text = "This    has    multiple    spaces    between    words"
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    # Verify spaces are normalized to single spaces
    assert chunks[0].data == "This has multiple spaces between words"
    # Verify no multiple spaces are preserved (as per BaseChunker behavior)
    assert "    " not in chunks[0].data

def test_space_normalization(chunker):
    # Test that various types of whitespace are normalized
    text = "This  has\tmultiple\n\rspaces\f\vbetween\t\t\twords"
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    assert chunks[0].data == "This has multiple spaces between words"
    
    # Verify no original whitespace characters remain
    assert "\t" not in chunks[0].data
    assert "\n" not in chunks[0].data
    assert "\r" not in chunks[0].data
    assert "\f" not in chunks[0].data
    assert "\v" not in chunks[0].data
    assert "  " not in chunks[0].data  # No double spaces

def test_chunking_preserves_single_spaces(chunker):
    # Test that single spaces between words are preserved
    text = "This has normal spaces between words"
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    assert chunks[0].data == "This has normal spaces between words"
    # Verify words are still separated by single spaces
    assert " " in chunks[0].data
    # Let's count the actual spaces:
    # "This[1]has[2]normal[3]spaces[4]between[5]words"
    assert chunks[0].data.count(" ") == 5  

def test_chunking_with_leading_trailing_spaces(chunker):
    # Test handling of leading and trailing spaces
    text = "   Leading and trailing spaces   "
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    # Verify leading and trailing spaces are trimmed
    assert chunks[0].data == "Leading and trailing spaces"

def test_chunking_with_newlines(chunker):
    # Test chunking text with newlines
    text = "Line 1\nLine 2\nLine 3\n" * 20
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    # Verify newlines are converted to spaces
    assert "\n" not in chunks[0].data

def test_large_text_performance():
    # Test performance with large text
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    large_text = "Sample text " * 1000
    
    chunks = chunker.chunk(large_text)
    assert len(chunks) > 0

def test_chunk_content_preservation(chunker):
    # Test that the original content is preserved across chunks
    text = "This is a specific test text that should be preserved across chunks."
    chunks = chunker.chunk(text)
    
    # Reconstruct text from chunks (considering overlap)
    reconstructed = chunks[0].data
    for i in range(1, len(chunks)):
        current_chunk = chunks[i].data
        overlap_size = chunker.chunk_overlap
        reconstructed += current_chunk[overlap_size:]
    
    # Remove any extra spaces that might have been added during chunking
    original_cleaned = ' '.join(text.split())
    reconstructed_cleaned = ' '.join(reconstructed.split())
    
    assert original_cleaned == reconstructed_cleaned
