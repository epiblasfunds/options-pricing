"""Validation helpers."""

from __future__ import annotations

import pandas as pd


def ensure_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error when expected columns are missing."""

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_non_empty_frame(frame: pd.DataFrame, message: str) -> None:
    """Raise when a frame is empty."""

    if frame.empty:
        raise ValueError(message)
