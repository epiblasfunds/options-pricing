"""Helpers for diagnosis aggregations."""

import numpy as np
import pandas as pd

from src.enums.data_enums import VolatilityDBEnum


def build_error_heatmap_frame(
    frame: pd.DataFrame,
    *,
    moneyness_bins: int = 12,
    maturity_bins: int = 12,
    maturity_floor: float = 0.0,
) -> pd.DataFrame:
    """Aggregate mean absolute error over explicit moneyness/maturity bins."""

    if frame.empty:
        return pd.DataFrame(
            columns=["moneyness_bin", "maturity_bin", "AbsoluteError"]
        )

    heatmap_frame = frame.dropna(
        subset=["Moneyness", str(VolatilityDBEnum.TIME_TO_EXPIRATION), "AbsoluteError"]
    ).copy()
    if heatmap_frame.empty:
        return pd.DataFrame(
            columns=["moneyness_bin", "maturity_bin", "AbsoluteError"]
        )

    heatmap_frame["moneyness_bin"] = pd.cut(
        heatmap_frame["Moneyness"],
        bins=_bin_edges(heatmap_frame["Moneyness"], bins=moneyness_bins),
        include_lowest=True,
    )
    heatmap_frame["maturity_bin"] = pd.cut(
        heatmap_frame[str(VolatilityDBEnum.TIME_TO_EXPIRATION)],
        bins=_bin_edges(
            heatmap_frame[str(VolatilityDBEnum.TIME_TO_EXPIRATION)],
            bins=maturity_bins,
            floor=maturity_floor,
        ),
        include_lowest=True,
    )
    return (
        heatmap_frame.groupby(["moneyness_bin", "maturity_bin"], observed=False)[
            "AbsoluteError"
        ]
        .mean()
        .reset_index()
    )


def _bin_edges(
    series: pd.Series,
    *,
    bins: int,
    floor: float | None = None,
) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if floor is not None:
        minimum = float(floor)
    if maximum <= minimum:
        maximum = minimum + 1.0
    return np.linspace(minimum, maximum, int(bins) + 1)
