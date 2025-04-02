import pytest
from unittest.mock import Mock
from flotorch_core.logger.logger import Logger
from flotorch_core.logger.logger_provider import LoggerProvider

def test_logger_singleton():
    """Ensures Logger is a singleton."""
    provider_mock = Mock(spec=LoggerProvider)
    logger1 = Logger(provider_mock)
    logger2 = Logger()
    
    assert logger1 is logger2  # Both instances should be the same

def test_logger_requires_provider_on_first_initialization():
    """Ensures Logger requires a provider during first initialization."""
    Logger._instance = None  # Reset singleton for testing
    with pytest.raises(ValueError, match="LoggerProvider must be provided for the first initialization."):
        Logger()

def test_logger_delegates_logging():
    """Ensures Logger delegates log calls to the provider."""
    provider_mock = Mock(spec=LoggerProvider)
    Logger._instance = None  # Reset singleton
    logger = Logger(provider_mock)

    logger.log("INFO", "Test message")
    provider_mock.log.assert_called_once_with("INFO", "Test message")

def test_logger_helper_methods():
    """Tests helper methods info, error, warning, debug."""
    provider_mock = Mock(spec=LoggerProvider)
    Logger._instance = None  # Reset singleton
    logger = Logger(provider_mock)

    logger.info("Info message")
    provider_mock.log.assert_called_with("INFO", "Info message")

    logger.error("Error message")
    provider_mock.log.assert_called_with("ERROR", "Error message")

    logger.warning("Warning message")
    provider_mock.log.assert_called_with("WARNING", "Warning message")

    logger.debug("Debug message")
    provider_mock.log.assert_called_with("DEBUG", "Debug message")

def test_logger_reset_singleton():
    """Ensures Logger can be reinitialized after resetting the singleton."""
    provider_mock1 = Mock(spec=LoggerProvider)
    provider_mock2 = Mock(spec=LoggerProvider)
    
    Logger._instance = None  # Reset singleton
    logger1 = Logger(provider_mock1)
    Logger._instance = None  # Reset singleton again
    logger2 = Logger(provider_mock2)
    
    assert logger1 is not logger2  # New instance should be created
    assert logger2.provider is provider_mock2

def test_logger_invalid_log_level():
    """Ensures Logger handles invalid log levels gracefully."""
    provider_mock = Mock(spec=LoggerProvider)
    Logger._instance = None  # Reset singleton
    logger = Logger(provider_mock)

    logger.log("INVALID", "Invalid log level message")
    provider_mock.log.assert_called_with("INVALID", "Invalid log level message")