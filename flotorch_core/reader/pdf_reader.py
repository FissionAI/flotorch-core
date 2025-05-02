import io

from PyPDF2 import PdfReader

from flotorch_core.storage.storage import StorageProvider


class PDFReader:
    """
    PDFReader is a class that reads PDF files from a storage provider.
    It uses the PyPDF2 library to extract text from the PDF files.
    """
    def __init__(self, storage_provider: StorageProvider):
        """
        Initializes the PDFReader with a storage provider.
        Args:
            storage_provider (StorageProvider): The storage provider to read PDF files from.
        """
        self.storage_provider = storage_provider

    def read_pdf(self, path: str) -> list[str]:
        """
        Reads a PDF file from the storage provider and extracts text from it.
        Args:
            path (str): The path to the PDF file in the storage provider.
        Returns:
            list[str]: A list of strings containing the extracted text from the PDF file.
        """
        text = []
        for data in self.storage_provider.read(path):
            if data is not None:
                text.append(self._read_pdf(data))
        return text

    def _read_pdf(self, data: bytes) -> str:
        """
        Reads a PDF file from bytes and extracts text from it.
        Args:
            data (bytes): The PDF file data in bytes.
        Returns:
            str: The extracted text from the PDF file.
        """
        stream = io.BytesIO(data)
        reader = PdfReader(stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
