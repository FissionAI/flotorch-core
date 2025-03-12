from abc import ABC, abstractmethod
import logging

class LoggerProvider(ABC):
    """
    Abstract base class for logger providers.
    """

    @abstractmethod
    def log(self, level: str, message: str) -> None:
        """
        Logs a message at the given level.
        Args:
            level (str): The logging level (e.g., 'INFO', 'ERROR').
            message (str): The message to log.
        """
        pass

    @abstractmethod
    def get_logger(self) -> logging.Logger:
        """
        Returns the underlying logger instance.
        """
        pass
