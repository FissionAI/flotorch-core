import io
import unittest
from unittest.mock import Mock, patch

import pytest
from PyPDF2 import PdfReader

from flotorch_core.storage.storage import StorageProvider
# Fix the import path to match your project structure
from flotorch_core.reader.pdf_reader import PDFReader


class TestPDFReader(unittest.TestCase):
    def setUp(self):
        # Create a mock storage provider
        self.mock_storage_provider = Mock(spec=StorageProvider)
        
        # Initialize the PDFReader with the mock storage provider
        self.pdf_reader = PDFReader(self.mock_storage_provider)
        
        # Sample PDF content - this would normally be binary PDF data
        # For testing purposes, we'll mock this later
        self.sample_pdf_data = b"mock pdf data"

    @patch('flotorch_core.reader.pdf_reader.PdfReader')
    def test_read_pdf_single_file(self, mock_pdf_reader_class):
        # Configure the mock storage provider to return our sample data
        self.mock_storage_provider.read.return_value = [self.sample_pdf_data]
        
        # Configure the mock PdfReader
        mock_pdf_reader = Mock()
        mock_pdf_reader_class.return_value = mock_pdf_reader
        
        # Create a mock page
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is sample text from a PDF page."
        
        # Set up the reader to return our mock page
        mock_pdf_reader.pages = [mock_page]
        
        # Call the method under test
        result = self.pdf_reader.read_pdf("test/path/to/pdf")
        
        # Assertions
        self.mock_storage_provider.read.assert_called_once_with("test/path/to/pdf")
        mock_pdf_reader_class.assert_called_once()
        self.assertEqual(result, ["This is sample text from a PDF page."])
        
        # Verify the BytesIO was created with our sample data
        # We can check this through the mock_pdf_reader_class call args
        args, _ = mock_pdf_reader_class.call_args
        self.assertIsInstance(args[0], io.BytesIO)
        # Reset the stream position to check content
        args[0].seek(0)
        self.assertEqual(args[0].read(), self.sample_pdf_data)
    
    @patch('flotorch_core.reader.pdf_reader.PdfReader')
    def test_read_pdf_multiple_files(self, mock_pdf_reader_class):
        # Configure the mock storage provider to return multiple files
        self.mock_storage_provider.read.return_value = [self.sample_pdf_data, self.sample_pdf_data]
        
        # Configure the mock PdfReader
        mock_pdf_reader = Mock()
        mock_pdf_reader_class.return_value = mock_pdf_reader
        
        # Create a mock page
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is sample text from a PDF page."
        
        # Set up the reader to return our mock page
        mock_pdf_reader.pages = [mock_page]
        
        # Call the method under test
        result = self.pdf_reader.read_pdf("test/path/to/pdfs")
        
        # Assertions
        self.mock_storage_provider.read.assert_called_once_with("test/path/to/pdfs")
        self.assertEqual(mock_pdf_reader_class.call_count, 2)
        self.assertEqual(result, ["This is sample text from a PDF page.", "This is sample text from a PDF page."])
    
    @patch('flotorch_core.reader.pdf_reader.PdfReader')
    def test_read_pdf_multiple_pages(self, mock_pdf_reader_class):
        # Configure the mock storage provider to return our sample data
        self.mock_storage_provider.read.return_value = [self.sample_pdf_data]
        
        # Configure the mock PdfReader
        mock_pdf_reader = Mock()
        mock_pdf_reader_class.return_value = mock_pdf_reader
        
        # Create mock pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content."
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content."
        
        # Set up the reader to return our mock pages
        mock_pdf_reader.pages = [mock_page1, mock_page2]
        
        # Call the method under test
        result = self.pdf_reader.read_pdf("test/path/to/pdf")
        
        # Assertions
        self.assertEqual(result, ["Page 1 content.Page 2 content."])
    
    @patch('flotorch_core.reader.pdf_reader.PdfReader')
    def test_read_pdf_empty_page(self, mock_pdf_reader_class):
        # Configure the mock storage provider to return our sample data
        self.mock_storage_provider.read.return_value = [self.sample_pdf_data]
        
        # Configure the mock PdfReader
        mock_pdf_reader = Mock()
        mock_pdf_reader_class.return_value = mock_pdf_reader
        
        # Create mock pages with one empty page
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page content."
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = None
        
        # Set up the reader to return our mock pages
        mock_pdf_reader.pages = [mock_page1, mock_page2]
        
        # Call the method under test
        result = self.pdf_reader.read_pdf("test/path/to/pdf")
        
        # Assertions
        self.assertEqual(result, ["Page content."])
    
    def test_read_pdf_empty_response(self):
        # Configure the mock storage provider to return None
        self.mock_storage_provider.read.return_value = [None]
        
        # Call the method under test
        result = self.pdf_reader.read_pdf("test/path/to/nonexistent")
        
        # Assertions
        self.assertEqual(result, [])
        self.mock_storage_provider.read.assert_called_once_with("test/path/to/nonexistent")

    @patch('flotorch_core.reader.pdf_reader.PdfReader')
    def test_read_pdf_unicode_characters(self, mock_pdf_reader_class):
        # Configure the mock storage provider to return our sample data
        self.mock_storage_provider.read.return_value = [self.sample_pdf_data]
        
        # Configure the mock PdfReader
        mock_pdf_reader = Mock()
        mock_pdf_reader_class.return_value = mock_pdf_reader
        
        # Create a mock page with various Unicode content including all requested examples
        unicode_text = (
            '"question": "¿Qué es la IA?", '
            '"answer": "Inteligencia Artificial 🤖", '
            '"chinese": "人工智能", '
            '"arabic": "الذكاء الاصطناعي", '
            '"hindi": "आर्टिफिशियल इंटेलिजेंस", '
            '"japanese": "人工知能", '
            '"emoji": "🔥🚀"'
        )
        
        mock_page = Mock()
        mock_page.extract_text.return_value = unicode_text
        
        # Set up the reader to return our mock page
        mock_pdf_reader.pages = [mock_page]
        
        # Call the method under test
        result = self.pdf_reader.read_pdf("test/path/to/unicode_pdf")
        
        # Assertions
        self.assertEqual(result, [unicode_text])
        
        # Verify each specific character set is preserved correctly
        self.assertIn("¿Qué es la IA?", result[0])  # Spanish with accents
        self.assertIn("Inteligencia Artificial 🤖", result[0])  # Spanish with robot emoji
        self.assertIn("人工智能", result[0])  # Chinese
        self.assertIn("الذكاء الاصطناعي", result[0])  # Arabic
        self.assertIn("आर्टिफिशियल इंटेलिजेंस", result[0])  # Hindi
        self.assertIn("人工知能", result[0])  # Japanese
        self.assertIn("🔥🚀", result[0])  # Emojis


if __name__ == "__main__":
    unittest.main()