import os
from typing import Any
from .config_provider import ConfigProvider

class EnvConfigProvider(ConfigProvider):
    """
    Configuration provider that fetches values from environment variables.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves the value of the environment variable or .env file.
        Args:
            key (str): The environment variable key.
            default (Any): The default value if the key is not found.
        Returns:
            Any: The value of the environment variable or the default.
        """
        return os.getenv(key, default)
