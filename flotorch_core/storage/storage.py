import logging
from abc import ABC, abstractmethod
from typing import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageProvider(ABC):
    """
    Abstract base class for storage
    """

    @abstractmethod
    def get_path(self, uri: str) -> str:
        """
        Gets the path from the URI.
        Args:
            uri (str): The URI to get the path from.
        Returns:
            str: The path extracted from the URI.
        """
        pass

    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """
        Writes data to the specified path.
        Args:
            path (str): The path to write the data to.
            data (bytes): The data to write.
        """
        pass

    @abstractmethod
    def read(self, path: str) -> Generator[bytes, None, None]:
        """
        Reads data from the specified path.
        Args:
            path (str): The path to read the data from.
        Returns:
            Generator[bytes, None, None]: A generator that yields the data read.
        """
        pass

    def read_as_string(self, path: str) -> Generator[str, None, None]:
        """
        Reads data from the specified path and yields it as a string.
        Args:
            path (str): The path to read the data from.
        Returns:
            Generator[str, None, None]: A generator that yields the data read as a string.
        """
        yield from (data.decode('utf-8') for data in self.read(path))
