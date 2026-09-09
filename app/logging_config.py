"""
Structured JSON logging for VittaBodh.
All log records are emitted as single-line JSON objects, making them
trivially parseable by Cloud Logging, Datadog, or any log aggregator.
"""
import json
import logging
import time


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Carry any structured extra fields attached by callers
        for key in ("uid", "latency_ms", "decision", "prediction_token", "feature", "z_score"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(app) -> None:
    """Attach a JSON handler to the Flask app logger and root logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    # Flask app logger
    app.logger.handlers = [handler]
    app.logger.setLevel(logging.INFO)
    app.logger.propagate = False

    # Root logger — only WARNING+ from third-party libs to reduce noise
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.WARNING)

    # VittaBodh package — INFO+
    vb_logger = logging.getLogger("app")
    vb_logger.setLevel(logging.INFO)
    vb_logger.propagate = True

    app.logger.info("VittaBodh structured logging initialised")
