from flotorch_core.logger.logger_provider import LoggerProvider

class Logger:
    """
    Main logger class that delegates logging to a logger provider.
    """

    _instance = None  # Singleton instance

    def __new__(cls, provider: LoggerProvider = None):
        """
        Create a singleton instance of the Logger class.
        Args:
            provider (LoggerProvider): The logger provider to use for logging.
        Returns:
            Logger: The singleton instance of the Logger class.
        """
        if cls._instance is None:
            if provider is None:
                raise ValueError("LoggerProvider must be provided for the first initialization.")
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.provider = provider
        return cls._instance

    def log(self, level: str, message: str) -> None:
        """
        Log a message at the specified level.
        Args:
            level (str): The logging level (e.g., "INFO", "ERROR", "WARNING", "DEBUG").
            message (str): The message to log.
        """
        self.provider.log(level, message)

    def info(self, message: str) -> None:
        """
        Log an informational message.
        Args:
            message (str): The message to log.
        """
        self.log("INFO", message)

    def error(self, message: str) -> None:
        """
        Log an error message.
        Args:
            message (str): The message to log.
        """
        self.log("ERROR", message)

    def warning(self, message: str) -> None:
        """
        Log a warning message.
        Args:
            message (str): The message to log.
        """
        self.log("WARNING", message)

    def debug(self, message: str) -> None:
        """
        Log a debug message.
        Args:
            message (str): The message to log.
        """
        self.log("DEBUG", message)
