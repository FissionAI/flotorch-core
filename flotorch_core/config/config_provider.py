from abc import ABC, abstractmethod
from typing import Any

class ConfigProvider(ABC):
    """
    Abstract base class for configuration providers.
    """
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves the configuration value for the given key.
        Args:
            key (str): The configuration key to retrieve.
            default (str): The default value when key not present
        Returns:
            Any: The value of the configuration key.
        """
        pass
