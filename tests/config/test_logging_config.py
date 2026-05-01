import logging

from src.config.logging_config.logging_config import LoggingConfig


def test_setup_logging_uses_explicit_level(monkeypatch):
    captured = {}

    def fake_dict_config(payload):
        captured["payload"] = payload

    monkeypatch.setattr(logging.config, "dictConfig", fake_dict_config)
    monkeypatch.setenv("LOG_LEVEL", "warning")

    LoggingConfig.setup_logging(level="debug")

    assert captured["payload"]["root"]["level"] == "DEBUG"
