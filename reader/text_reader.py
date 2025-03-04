import io
from typing import List
from storage.storage import StorageProvider

class TXTReader:
    """A class to read text files using a storage provider."""

    def __init__(self, storage_provider: StorageProvider):
        """Initialize TXTReader with a storage provider.
        
        Args:
            storage_provider: Storage provider instance to read files
        """
        self.storage_provider = storage_provider

    def read_txt(self, path: str) -> List[str]:
        """Read text file(s) from the given path.
        
        Args:
            path: Path to the text file(s)
            
        Returns:
            List of decoded text strings
        """
        text = []
        for data in self.storage_provider.read(path):
            if data is not None:
                text.append(self._read_txt(data))
        return text

    def _read_txt(self, data: bytes) -> str:
        """Decode bytes data to string using various encodings.
        
        Args:
            data: Bytes data to decode
            
        Returns:
            Decoded string or empty string if decoding fails
        """
        try:
            # Try UTF-8 first as it's most common
            return data.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, try other common encodings
            encodings = ['latin-1', 'ascii', 'utf-16']
            for encoding in encodings:
                try:
                    return data.decode(encoding)
                except UnicodeDecodeError:
                    continue
            # If all decoding attempts fail, return an empty string or handle the error as needed
            return ""

