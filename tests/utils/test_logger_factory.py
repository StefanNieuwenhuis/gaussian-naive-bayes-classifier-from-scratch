import logging
import pytest

from utils.logging_factory import LoggerFactory


class TestLoggerFactory:
    """
    Tests for LoggerFactory static utils class
    """
    def test_get_logger_returns_logger_instance(self) -> None:
        """
        It should create an instance of LoggerFactory
        """

        # Arrange & Act
        logger = LoggerFactory.get_logger("test_logger")

        # Assert
        assert isinstance(logger, logging.Logger)

    def test_get_logger_reuses_logger_by_name(self) -> None:
        """
        Multiple LoggerFactory instances should be the same
        """

        # Arrange & Act
        logger1 = LoggerFactory.get_logger("reused_logger")
        logger2 = LoggerFactory.get_logger("reused_logger")

        # Assert
        assert logger1 is logger2


    def test_logger_has_expected_handler_and_formatter(self) -> None:
        """
        It should contain handlers and formatters
        """

        # Arrange & Act
        logger = LoggerFactory.get_logger("format_test_logger")
        handler = logger.handlers[0]
        formatter = handler.formatter
        formatted = formatter.format(logging.LogRecord(
            name="format_test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="test message",
            args=(),
            exc_info=None
        ))

        # Assert
        assert logger.handlers, "Logger should have at least one handler"
        assert isinstance(handler, logging.StreamHandler)
        assert formatter is not None
        assert "format_test_logger" in formatted
        assert "test message" in formatted


    def test_logger_does_not_duplicate_handlers(self) -> None:
        """
        It should not duplicate handlers
        """

        # Arrange & Act
        logger = LoggerFactory.get_logger("no_duplicate_handler")
        original_handler_count = len(logger.handlers)

        # Force retrieval again
        logger = LoggerFactory.get_logger("no_duplicate_handler")

        # Assert
        assert len(logger.handlers) == original_handler_count

    def test_logger_outputs_to_stream(self, caplog) -> None:
        """
        It should output to stream
        """

        # Arrange & Act
        logger = LoggerFactory.get_logger("stream_test")
        with caplog.at_level(logging.INFO):
            logger.info("Hello stream")

        # Assert
        assert "Hello stream" in caplog.text
