from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DBStorage(ABC):
    """
    This class is responsible for storing the data in the database.
    """

    @abstractmethod
    def read(self, key) -> dict:
        """
        Read the data from the database.
        Args:
            key: The key to read the data from the database."""
        pass

    @abstractmethod
    def write(self, item: dict):
        """
        Write the data to the database.
        Args:
            item: The item to write to the database.
        """
        pass

    def bulk_write(self, items: List[dict]):
        """
        Write multiple items to the database.
        Args:
            items: The items to write to the database.
        """
        for item in items:
            self.write(item)
