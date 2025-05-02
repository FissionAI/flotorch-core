from .logger_provider import LoggerProvider
import logging

class ConsoleLoggerProvider(LoggerProvider):
    """
    Logger provider that logs messages to the console.
    """

    def __init__(self, name: str = "default"):
        """
        Initializes the ConsoleLoggerProvider with a given name.
        Args:
            name (str): The name of the logger.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:  
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: str, message: str) -> None:
        """
        Logs a message at the specified level.
        Args:
            level (str): The logging level (e.g., 'info', 'debug', 'warning', 'error', 'critical').
            message (str): The message to log.
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

    def get_logger(self) -> logging.Logger:
        """
        Returns the logger instance.
        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger
