from flotorch_core.logger.logger import Logger
from flotorch_core.logger.console_logger_provider import ConsoleLoggerProvider

def get_logger():
    """
    Returns a singleton logger instance.
    """
    provider = ConsoleLoggerProvider()
    return Logger(provider)
