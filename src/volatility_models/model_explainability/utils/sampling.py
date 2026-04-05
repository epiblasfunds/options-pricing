"""Sampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_frame(
    frame: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Return a deterministic sample if the frame is larger than the requested size."""

    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=random_state).sort_index()


def quantile_grid(series: pd.Series, points: int) -> list[float]:
    """Stable quantile grid for charting."""

    clean = series.dropna().astype(float)
    if clean.empty:
        return []
    quantiles = np.linspace(0.05, 0.95, points)
    values = clean.quantile(quantiles).round(8).tolist()
    return sorted({float(value) for value in values})
