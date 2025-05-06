import logging
import pytest
from flotorch_core.logger.console_logger_provider import ConsoleLoggerProvider

def test_console_logger_initialization():
    """Test that ConsoleLoggerProvider initializes correctly."""
    logger_provider = ConsoleLoggerProvider(name="test_logger")
    logger = logger_provider.get_logger()

    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1  # Ensure only one handler is added
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_console_logger_does_not_add_duplicate_handlers():
    """Ensure multiple instances do not add duplicate handlers."""
    logger_provider1 = ConsoleLoggerProvider(name="test_logger")
    logger_provider2 = ConsoleLoggerProvider(name="test_logger")
    
    logger = logger_provider1.get_logger()
    assert len(logger.handlers) == 1  # Should still be one handler

@pytest.mark.parametrize("log_level", ["debug", "info", "warning", "error", "critical"])
def test_console_logger_log_levels(caplog, log_level):
    """Test logging at different levels."""
    logger_provider = ConsoleLoggerProvider(name="test_logger")
    
    # Set the logger's level to the lowest level (DEBUG) to capture all messages
    logger_provider.get_logger().setLevel(logging.DEBUG)
    
    with caplog.at_level(getattr(logging, log_level.upper())):
        logger_provider.log(log_level, f"Test {log_level} message")

    assert f"Test {log_level} message" in caplog.text

def test_console_logger_invalid_log_level(caplog):
    """Test passing an invalid log level (should default to info)."""
    logger_provider = ConsoleLoggerProvider(name="test_logger")
    
    with caplog.at_level(logging.INFO):
        logger_provider.log("invalid_level", "Fallback to info message")

    assert "Fallback to info message" in caplog.text

def test_console_logger_custom_name():
    logger_provider = ConsoleLoggerProvider(name="custom_logger")
    logger = logger_provider.get_logger()
    assert logger.name == "custom_logger"

def test_console_logger_formatter():
    logger_provider = ConsoleLoggerProvider(name="test_logger")
    logger = logger_provider.get_logger()
    handler = logger.handlers[0]
    assert isinstance(handler.formatter, logging.Formatter)
    assert handler.formatter._fmt == "%(asctime)s - %(levelname)s - %(message)s"