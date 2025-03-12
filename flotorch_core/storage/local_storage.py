import logging
import os
from typing import Generator

from .storage import StorageProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalStorageProvider(StorageProvider):
    """
    Local storage provider
    """

    def __init__(self):
        """
        Initializes the LocalStorageProvider class.
        """
        super().__init__()

    def get_path(self, uri: str) -> str:
        return uri.split("://")[1]

    def write(self, path: str, data: bytes) -> None:
        """
        Writes data to the specified path in local storage.
        Args:
            path (str): The path to write the data to in local storage.
            data (bytes): The data to write to local storage.
        """
        logger.info(f'Writing data to local storage: {data}')
        if os.path.isdir(path):
            path = os.path.join(path, 'tmp.data')
        with open(path, 'wb') as file:
            file.write(data)

    def read(self, path) -> Generator[bytes, None, None]:
        """
        Reads data from the specified path in local storage.
        Args:
            path (str): The path to read the data from in local storage.
        Returns:
            Generator[bytes, None, None]: A generator that yields the data read from local storage.
        """
        logger.info('Reading data from local storage')
        if os.path.isdir(path):
            yield from self._read_directory(path)
        else:
            with open(path, "rb") as file:
                yield file.read()

    def _read_directory(self, path):
        """
        Reads all files in a directory.
        Args:
            path (str): The path to the directory to read.
        Returns:
            Generator[bytes, None, None]: A generator that yields the data read from each file in the directory.
        """
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                with open(path, "rb") as file:
                    return file.read()
