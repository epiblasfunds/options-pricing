"""Shared styling helpers for Plotly figures."""

import math
from collections.abc import Iterable

from plotly.colors import sample_colorscale

VOLATILITY_COLORSCALE = "Cividis"
ERROR_COLORSCALE = "YlOrRd"
STANDARD_TEMPLATE = "plotly_white"
STANDARD_MARGIN = dict(l=30, r=20, t=48, b=30)
HOVERLABEL_STYLE = dict(
    bgcolor="#102542",
    bordercolor="#7aa5d2",
    font=dict(color="#f8fbff", size=12),
    namelength=-1,
)


def safe_color_range(values: Iterable[float]) -> tuple[float, float]:
    numeric_values = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_value):
            continue
        numeric_values.append(numeric_value)
    if not numeric_values:
        return (0.0, 1.0)
    minimum = min(numeric_values)
    maximum = max(numeric_values)
    if math.isclose(minimum, maximum, rel_tol=1e-9, abs_tol=1e-9):
        padding = max(abs(minimum) * 0.05, 1e-6)
        return (minimum - padding, maximum + padding)
    return (minimum, maximum)


def sample_scale_color(
    value: float,
    value_range: tuple[float, float],
    colorscale: str = VOLATILITY_COLORSCALE,
) -> str:
    lower, upper = value_range
    if math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-9):
        normalized = 0.5
    else:
        normalized = (float(value) - lower) / (upper - lower)
    normalized = min(1.0, max(0.0, normalized))
    return str(sample_colorscale(colorscale, [normalized])[0])


def truncated_colorscale(
    colorscale: str,
    start: float = 0.0,
    end: float = 1.0,
    steps: int = 8,
) -> list[list[object]]:
    start = min(1.0, max(0.0, float(start)))
    end = min(1.0, max(start, float(end)))
    steps = max(2, int(steps))
    positions = [index / (steps - 1) for index in range(steps)]
    sampled = sample_colorscale(
        colorscale,
        [start + (end - start) * position for position in positions],
    )
    return [[position, color] for position, color in zip(positions, sampled)]
