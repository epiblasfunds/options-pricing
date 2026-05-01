import pandas as pd

from src.dashboard.utils.sampling import quantile_grid
from src.dashboard.utils.sampling import sample_frame


def test_sample_frame_is_deterministic_only_when_needed():
    frame = pd.DataFrame({"value": range(6)}, index=[10, 11, 12, 13, 14, 15])

    sampled = sample_frame(frame, max_rows=3, random_state=7)
    full = sample_frame(frame, max_rows=10, random_state=7)

    assert sampled.index.tolist() == [10, 13, 15]
    assert full.equals(frame)


def test_quantile_grid_returns_unique_sorted_values():
    series = pd.Series([1, 1, 2, 3, 4, 4, None])

    grid = quantile_grid(series, points=5)

    assert grid == sorted(grid)
    assert len(grid) == len(set(grid))
    assert quantile_grid(pd.Series([None, None]), points=5) == []
