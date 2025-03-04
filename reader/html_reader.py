import io
from bs4 import BeautifulSoup
from typing import List
from storage.storage import StorageProvider

class HTMLReader:
    def __init__(self, storage_provider: StorageProvider):
        self.storage_provider = storage_provider

    def read_html(self, path: str) -> List[str]:
        text = []
        for data in self.storage_provider.read(path):
            if data is not None:
                text.append(self._read_html(data))
        return text

    def _read_html(self, data: bytes) -> str:
        try:
            html_string = data.decode('utf-8')
            soup = BeautifulSoup(html_string, 'html.parser')
            for element in soup(["script", "style"]):
                element.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return ' '.join(text.split())
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return ""
