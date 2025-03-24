import logging
import pytest
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.logger.logger import Logger
from flotorch_core.logger.console_logger_provider import ConsoleLoggerProvider

def test_get_logger_returns_logger_instance():
    """Test that get_logger() returns a Logger instance."""
    logger = get_logger()
    assert isinstance(logger, Logger)

def test_get_logger_uses_console_logger_provider():
    """Test that get_logger() initializes with ConsoleLoggerProvider."""
    logger = get_logger()
    assert isinstance(logger.provider, ConsoleLoggerProvider)

def test_get_logger_singleton():
    """Test that get_logger() returns the same instance (singleton)."""
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2  # Ensure both are the same instance

def test_get_logger_logs_messages(caplog):
    """Test that logging through get_logger() works."""
    logger = get_logger()
    
    with caplog.at_level(logging.INFO):
        logger.log("info", "Test singleton logger message")
    
    assert "Test singleton logger message" in caplog.text
