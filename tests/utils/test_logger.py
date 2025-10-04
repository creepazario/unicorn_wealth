import json
import logging
from io import StringIO
import sys


from utils.logger import setup_logger


class DummyHandler(logging.Handler):
    """A no-op logging handler used to replace TimedRotatingFileHandler during tests."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - no-op
        pass


def test_logger_produces_json_output(mocker):
    # Patch TimedRotatingFileHandler to avoid file I/O
    mocker.patch(
        "logging.handlers.TimedRotatingFileHandler",
        lambda *args, **kwargs: DummyHandler(),
    )

    # Patch Path.mkdir to avoid filesystem writes
    mocker.patch("pathlib.Path.mkdir", lambda *args, **kwargs: None)

    # Capture stdout with StringIO so StreamHandler writes here
    buffer = StringIO()
    mocker.patch.object(sys, "stdout", buffer)

    # Create a dummy settings-like object with required attributes
    class DummySettings:
        log_level = "INFO"
        log_file_path = "dummy/path.log"

    # Configure logger
    logger = setup_logger(DummySettings())

    # Emit a debug (should be filtered) and an info message
    logger.debug("should not appear")
    logger.info("test message")

    # Get captured output
    output = buffer.getvalue().strip()
    # There may be multiple lines due to prior handlers if any;
    # consider only non-empty lines
    lines = [line for line in output.splitlines() if line.strip()]

    # We expect exactly one line (INFO) because DEBUG is suppressed at INFO level
    assert len(lines) == 1, f"Expected 1 log line, got {len(lines)}: {lines}"

    # Validate JSON format
    parsed = json.loads(lines[0])  # should not raise

    # Validate the message content
    assert parsed.get("message") == "test message"

    # Validate common structured keys exist
    for key in ("timestamp", "level", "name", "message"):
        assert key in parsed
