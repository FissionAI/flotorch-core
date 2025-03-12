from flotorch_core.logger.logger_provider import LoggerProvider

class Logger:
    """
    Main logger class that delegates logging to a logger provider.
    """

    _instance = None  # Singleton instance

    def __new__(cls, provider: LoggerProvider = None):
        if cls._instance is None:
            if provider is None:
                raise ValueError("LoggerProvider must be provided for the first initialization.")
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.provider = provider
        return cls._instance

    def log(self, level: str, message: str) -> None:
        self.provider.log(level, message)

    def info(self, message: str) -> None:
        self.log("INFO", message)

    def error(self, message: str) -> None:
        self.log("ERROR", message)

    def warning(self, message: str) -> None:
        self.log("WARNING", message)

    def debug(self, message: str) -> None:
        self.log("DEBUG", message)
