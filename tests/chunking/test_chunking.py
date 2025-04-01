import pytest
from flotorch_core.chunking.chunking import Chunk, BaseChunker
from typing import List


def test_chunk_initialization():
    # Test basic chunk creation
    data = "test data"
    chunk = Chunk(data)

    assert chunk.data == data
    assert chunk.child_data is None
    assert isinstance(chunk.id, str)  # Verify UUID was created

def test_chunk_add_child():
    # Test adding child data to chunk
    chunk = Chunk("parent data")
    child_data = "child data"
    
    chunk.add_child(child_data)
    
    assert chunk.child_data == [child_data]
    
    # Test adding multiple children
    second_child = "second child"
    chunk.add_child(second_child)
    assert chunk.child_data == [child_data, second_child]

def test_chunk_str_representation():
    # Test string representation of chunk
    chunk = Chunk("parent")
    assert str(chunk) == "Parent Chunk: parent, Chunk: None"
    
    chunk.add_child("child")
    assert str(chunk) == "Parent Chunk: parent, Chunk: ['child']"

# Create a concrete implementation of BaseChunker for testing
class TestChunker(BaseChunker):
    def chunk(self, data: str) -> List[Chunk]:
        cleaned_data = self._clean_data(data)
        return [Chunk(cleaned_data)]

@pytest.fixture
def chunker():
    return TestChunker()

def test_base_chunker_initialization(chunker):
    assert chunker.space == ' '
    assert chunker.separators == [' ', '\t', '\n', '\r', '\f', '\v']
    assert chunker.tokens_per_charecter == 4

def test_clean_data(chunker):
    # Test cleaning data with different separators
    input_text = "Hello\tWorld\nNew\rLine\fTest\vText"
    expected = "Hello World New Line Test Text"
    
    cleaned = chunker._clean_data(input_text)
    assert cleaned == expected

def test_chunk_list(chunker):
    # Test chunking a list of texts
    input_texts = ["Hello World", "Test Text"]
    chunks = chunker.chunk_list(input_texts)
    
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].data == "Hello World"
    assert chunks[1].data == "Test Text"

def test_base_chunker_is_abstract():
    # Verify that BaseChunker cannot be instantiated directly
    with pytest.raises(TypeError):
        BaseChunker()

def test_chunker_with_empty_input(chunker):
    # Test handling empty input
    chunks = chunker.chunk("")
    assert len(chunks) == 1
    assert chunks[0].data == ""

def test_chunker_with_none_input(chunker):
    # Test handling None input
    with pytest.raises(AttributeError):
        chunker.chunk(None)

def test_chunk_list_with_empty_list(chunker):
    # Test chunking empty list
    chunks = chunker.chunk_list([])
    assert chunks == []

def test_chunk_list_with_none_elements(chunker):
    # Test chunking list with None elements
    with pytest.raises(AttributeError):
        chunker.chunk_list([None])

def test_clean_data_with_multiple_spaces(chunker):
    # Test cleaning data with multiple consecutive spaces
    input_text = "Hello    World  Test"
    cleaned = chunker._clean_data(input_text)
    assert cleaned == "Hello    World  Test"  # Should preserve multiple spaces

def test_clean_data_with_mixed_separators(chunker):
    # Test cleaning data with mixed separators
    input_text = "Hello\t\nWorld\r\f\vTest"
    cleaned = chunker._clean_data(input_text)
    assert cleaned == "Hello  World   Test"

# Test edge cases
def test_chunk_with_special_characters(chunker):
    # Test chunking text with special characters
    input_text = "Hello! @#$%^&*()"
    chunks = chunker.chunk(input_text)
    assert chunks[0].data == "Hello! @#$%^&*()"

def test_chunk_with_unicode_characters(chunker):
    # Test chunking text with unicode characters
    input_text = "Hello 世界"
    chunks = chunker.chunk(input_text)
    assert chunks[0].data == "Hello 世界"

def test_chunk_list_with_mixed_content(chunker):
    # Test chunking list with mixed content types
    input_texts = ["Hello", "World\tTest", "Line\nBreak"]
    chunks = chunker.chunk_list(input_texts)
    
    assert len(chunks) == 3
    assert chunks[0].data == "Hello"
    assert chunks[1].data == "World Test"
    assert chunks[2].data == "Line Break"

# Performance test for large inputs
def test_large_input_performance(chunker):
    # Test performance with large input
    large_text = "word " * 1000
    chunks = chunker.chunk(large_text)
    assert len(chunks) > 0

# Test chunk ID uniqueness
def test_chunk_id_uniqueness():
    # Test that each chunk gets a unique ID
    chunk1 = Chunk("data1")
    chunk2 = Chunk("data2")
    assert chunk1.id != chunk2.id
