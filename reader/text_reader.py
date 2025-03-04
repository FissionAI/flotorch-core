import io
from typing import List
from storage.storage import StorageProvider

class TXTReader:
    def __init__(self, storage_provider: StorageProvider):
        self.storage_provider = storage_provider

    def read_txt(self, path: str) -> List[str]:
        text = []
        for data in self.storage_provider.read(path):
            if data is not None:
                text.append(self._read_txt(data))
        return text

    def _read_txt(self, data: bytes) -> str:
        try:
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

