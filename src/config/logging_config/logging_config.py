import logging
import logging.config
import os

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


class LoggingConfig:
    @staticmethod
    def setup_logging(level: str | None = None) -> None:
        """
        Configure logging once.
        - level: override by env or param (INFO, DEBUG, etc.)
        """
        cfg = dict(LOGGING_CONFIG)

        # Log level from environment or parameter
        final_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
        cfg["root"]["level"] = final_level

        logging.config.dictConfig(cfg)
