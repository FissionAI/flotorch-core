import pytest
from flotorch_core.chunking.hierarical_chunking import HieraricalChunker
from flotorch_core.chunking.chunking import Chunk

@pytest.fixture
def chunker():
    return HieraricalChunker(chunk_size=100, chunk_overlap=20, parent_chunk_size=500)

def test_hierarchical_chunker_initialization():
    chunker = HieraricalChunker(chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
    assert chunker.chunk_size == 400  # 100 * tokens_per_character (4)
    assert chunker.chunk_overlap == 80  # 20 tokens * 4
    assert chunker.parent_chunk_size == 500 * chunker.tokens_per_charecter

def test_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        HieraricalChunker(chunk_size=0, chunk_overlap=20, parent_chunk_size=500)
    
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        HieraricalChunker(chunk_size=-1, chunk_overlap=20, parent_chunk_size=500)

def test_invalid_chunk_overlap():
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        HieraricalChunker(chunk_size=100, chunk_overlap=100, parent_chunk_size=500)
    
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        HieraricalChunker(chunk_size=100, chunk_overlap=150, parent_chunk_size=500)

def test_chunk_empty_input(chunker):
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        chunker.chunk("")

def test_chunk_none_input(chunker):
    with pytest.raises(ValueError, match="Input text cannot be empty or None"):
        chunker.chunk(None)

def test_basic_hierarchical_chunking(chunker):
    text = """First paragraph with some content.

    Second paragraph with different content.
    
    Third paragraph with more content."""
    
    chunks = chunker.chunk(text)
    
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert len(chunks) > 0
    
    # Check if paragraphs are separated into different chunks
    assert any("First paragraph" in chunk.data for chunk in chunks)
    assert any("Second paragraph" in chunk.data for chunk in chunks)
    assert any("Third paragraph" in chunk.data for chunk in chunks)

def test_hierarchical_structure(chunker):
    text = """Chapter 1. Introduction
    First section content.
    More content here.

    Chapter 2. Main Content
    Second section content.
    Additional content here.

    Chapter 3. Conclusion
    Final section content."""
    
    chunks = chunker.chunk(text)
    
    # Verify chunks maintain hierarchical structure
    parent_chunks = [chunk for chunk in chunks if chunk.child_data is not None]
    assert len(parent_chunks) > 0
    
    # Verify child chunks exist
    for parent in parent_chunks:
        assert isinstance(parent.child_data, list)
        assert len(parent.child_data) > 0

def test_chunk_with_nested_content(chunker):
    text = """Main Section:
    - Subsection 1
      * Point A
      * Point B
    - Subsection 2
      * Point C
      * Point D"""
    
    chunks = chunker.chunk(text)
    
    # Verify main section is captured
    assert any("Main Section" in chunk.data for chunk in chunks)
    
    # Verify subsections are captured in child chunks
    parent_chunks = [chunk for chunk in chunks if chunk.child_data is not None]
    for parent in parent_chunks:
        if "Main Section" in parent.data:
            # Fix: Access the data attribute of child Chunk objects
            assert any("Subsection" in child.data for child in parent.child_data)

def test_nested_chunk_structure(chunker):
    text = """Main Section:
    - Subsection 1
      * Point A
      * Point B
    - Subsection 2
      * Point C
      * Point D"""
    
    chunks = chunker.chunk(text)
    
    # Find the main section chunk
    main_section_chunks = [chunk for chunk in chunks if "Main Section" in chunk.data]
    assert len(main_section_chunks) > 0
    
    main_chunk = main_section_chunks[0]
    assert main_chunk.child_data is not None
    
    # Verify child chunks are Chunk objects
    assert all(isinstance(child, Chunk) for child in main_chunk.child_data)
    
    # Verify subsection content in child chunks
    subsection_contents = [child.data for child in main_chunk.child_data]
    assert any("Subsection 1" in content for content in subsection_contents)
    assert any("Subsection 2" in content for content in subsection_contents)


def test_chunking_with_different_sizes():
    small_chunker = HieraricalChunker(chunk_size=50, chunk_overlap=10, parent_chunk_size=200)
    large_chunker = HieraricalChunker(chunk_size=200, chunk_overlap=30, parent_chunk_size=500)
    
    text = """Section 1
    Content for section 1.
    More content.

    Section 2
    Content for section 2.
    More content."""
    
    small_chunks = small_chunker.chunk(text)
    large_chunks = large_chunker.chunk(text)
    
    # Smaller chunk size should result in more chunks
    assert len(small_chunks) >= len(large_chunks)

def test_chunking_with_special_characters(chunker):
    text = """Section !@#$
    Content with @#$ characters.

    Section &*()
    More !@#$ content."""
    
    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    assert any("!@#$" in chunk.data for chunk in chunks)

def test_chunking_with_unicode(chunker):
    text = """Section 你好
    Content with unicode 世界.

    Section αβγ
    More 👋🌍 content."""
    
    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    assert any("你好" in chunk.data for chunk in chunks)
    assert any("αβγ" in chunk.data for chunk in chunks)

def test_large_document_chunking():
    chunker = HieraricalChunker(chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
    # Create a large document with multiple sections
    sections = []
    for i in range(10):
        sections.append(f"""Section {i}
        This is content for section {i}.
        It contains multiple lines.
        Each section has similar structure.""")
    
    text = "\n\n".join(sections)
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    # Verify sections are properly separated
    assert any(f"Section 0" in chunk.data for chunk in chunks)
    assert any(f"Section 9" in chunk.data for chunk in chunks)

def test_chunk_overlap_consistency(chunker):
    text = """Section 1
    This is a long section with enough content to test overlap.
    It continues for multiple lines to ensure we have overlap.

    Section 2
    This is another section that should overlap with the previous one.
    It also has multiple lines of content."""
    
    chunks = chunker.chunk(text)
    
    # Check if consecutive chunks have overlapping content
    for i in range(len(chunks) - 1):
        if chunks[i].child_data and chunks[i + 1].child_data:
            current_last_child = chunks[i].child_data[-1]
            next_first_child = chunks[i + 1].child_data[0]
            # Verify some content overlaps between chunks
            assert any(word in next_first_child for word in current_last_child.split())


def test_hierarchical_structure_preservation(chunker):
    # Create a longer text with multiple chapters to ensure proper chunking
    text = """Chapter 1
    This is the first chapter with substantial content.
    Section 1.1
    Content for section 1.1
    Adding more content to ensure proper chunking.
    This section contains detailed information.
    More lines to increase the content size.

    Section 1.2
    Content for section 1.2
    This section also needs sufficient content.
    Adding multiple lines of text here.
    Making sure we have enough content.

    Chapter 2
    The second chapter needs substantial content too.
    Section 2.1
    Content for section 2.1
    Adding more detailed information here.
    This section contains important details.
    Additional content to ensure proper size.

    Section 2.2
    More content for this section.
    Ensuring we have enough text for proper chunking.
    Adding several lines of content here.
    Making the section sufficiently large.""" * 2  # Multiply content to ensure enough text
    
    chunks = chunker.chunk(text)
    
    # First, let's check the overall structure
    assert len(chunks) > 0
    
    # Find chunks containing chapter content
    chapter_texts = [chunk.data for chunk in chunks]
    assert any("Chapter 1" in text for text in chapter_texts)
    assert any("Chapter 2" in text for text in chapter_texts)
    
    # Verify sections exist in child chunks
    sections_found = set()
    for chunk in chunks:
        if chunk.child_data:
            for child in chunk.child_data:
                if "Section" in child.data:
                    sections_found.add(child.data.split('\n')[0].strip())
    
    # Verify we found sections from both chapters
    assert any("Section 1" in section for section in sections_found)
    assert any("Section 2" in section for section in sections_found)

def test_basic_chapter_structure(chunker):
    # Test with minimal but sufficient content
    text = """Chapter 1
    This is chapter one content.
    More content for chapter one.
    Additional content here.
    
    Chapter 2
    This is chapter two content.
    More content for chapter two.
    Additional content here.""" * 3  # Multiply to ensure enough content
    
    chunks = chunker.chunk(text)
    
    # Verify both chapters are present in the chunks
    chapter_content = ' '.join(chunk.data for chunk in chunks)
    assert "Chapter 1" in chapter_content
    assert "Chapter 2" in chapter_content

def test_section_distribution(chunker):
    # Test how sections are distributed across chunks
    text = """Chapter 1
    Introduction text that needs to be substantial.
    We need enough content to ensure proper chunking.
    Section 1.1
    Content for section one point one.
    More content here to make the section larger.
    Adding several lines of meaningful content.
    This helps ensure proper chunking behavior.
    
    Section 1.2
    Content for section one point two.
    Additional content here to make this section substantial.
    More lines of text to increase section size.
    Making sure we have enough content.
    
    Chapter 2
    Chapter two introduction with substantial content.
    This chapter also needs to be long enough.
    Section 2.1
    Content for section two point one.
    More content here to ensure proper chunking.
    Adding multiple lines of meaningful text.
    This section needs to be large enough.""" * 3  # Multiply content for sufficient size
    
    chunks = chunker.chunk(text)
    
    # Debug information
    print("\nDebug: Chunks structure:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"Main content: {chunk.data[:100]}...")  # Print first 100 chars
        if chunk.child_data:
            print("Child chunks:")
            for j, child in enumerate(chunk.child_data):
                print(f"  Child {j}: {child.data[:100]}...")

    # Collect all section numbers using a more robust approach
    sections = set()
    for chunk in chunks:
        # Check main chunk content
        if "Section" in chunk.data:
            section_lines = [line.strip() for line in chunk.data.split('\n') if "Section" in line]
            for line in section_lines:
                if "Section" in line:
                    try:
                        section_num = line.split("Section")[1].strip().split()[0]
                        sections.add(section_num)
                    except IndexError:
                        continue

        # Check child chunks
        if chunk.child_data:
            for child in chunk.child_data:
                if "Section" in child.data:
                    section_lines = [line.strip() for line in child.data.split('\n') if "Section" in line]
                    for line in section_lines:
                        try:
                            section_num = line.split("Section")[1].strip().split()[0]
                            sections.add(section_num)
                        except IndexError:
                            continue

    print("\nDebug: Found sections:", sections)

    # Verify we have sections from both chapters
    assert len(sections) > 0, "No sections found in the chunks"
    chapter1_sections = [s for s in sections if s.startswith("1.")]
    chapter2_sections = [s for s in sections if s.startswith("2.")]
    
    assert len(chapter1_sections) > 0, "No sections from Chapter 1 found"
    assert len(chapter2_sections) > 0, "No sections from Chapter 2 found"
    
    # Print more specific information about what was found
    print("\nFound Chapter 1 sections:", chapter1_sections)
    print("Found Chapter 2 sections:", chapter2_sections)

def test_basic_section_identification(chunker):
    """A simpler test to verify basic section identification"""
    text = """Chapter 1
    Introduction
    Section 1.1
    Basic content for section one point one.
    
    Chapter 2
    Introduction
    Section 2.1
    Basic content for section two point one.""" * 5  # Multiply for sufficient content
    
    chunks = chunker.chunk(text)
    
    # Print chunk structure for debugging
    print("\nChunk structure:")
    for chunk in chunks:
        print(f"\nMain chunk: {chunk.data[:50]}...")
        if chunk.child_data:
            for child in chunk.child_data:
                print(f"  Child: {child.data[:50]}...")
    
    # Verify sections are present in the content
    all_content = ' '.join(chunk.data for chunk in chunks)
    assert "Section 1.1" in all_content, "Section 1.1 not found in content"
    assert "Section 2.1" in all_content, "Section 2.1 not found in content"

