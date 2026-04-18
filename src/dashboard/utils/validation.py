"""Validation helpers."""

import pandas as pd


def ensure_non_empty_frame(frame: pd.DataFrame, message: str) -> None:
    """Raise when a frame is empty."""

    if frame.empty:
        raise ValueError(message)
