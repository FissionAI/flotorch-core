import io
from bs4 import BeautifulSoup
import io
from typing import List
from storage.storage import StorageProvider

class HTMLReader:
    """Class to read and extract text content from HTML files"""
    
    def __init__(self, storage_provider: StorageProvider):
        """Initialize HTMLReader with storage provider
        
        Args:
            storage_provider: Storage provider instance to read files
        """
        self.storage_provider = storage_provider

    def read_html(self, path: str) -> List[str]:
        """Read HTML files and extract text content
        
        Args:
            path: Path to HTML file(s)
            
        Returns:
            List of extracted text strings from HTML files
        """
        text = []
        for data in self.storage_provider.read(path):
            if data is not None:
                text.append(self._read_html(data))
        return text

    def _read_html(self, data: bytes) -> str:
        """Extract text content from HTML data
        
        Args:
            data: HTML content as bytes
            
        Returns:
            Extracted text as string with whitespace normalized
        """
        try:
            # Decode bytes to string
            html_string = data.decode('utf-8')
            
            # Parse HTML
            soup = BeautifulSoup(html_string, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()
                
            # Extract text with normalized whitespace
            text = soup.get_text(separator=' ', strip=True)
            return ' '.join(text.split())
            
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return ""
