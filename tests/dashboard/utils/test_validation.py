import pandas as pd
import pytest

from src.dashboard.utils.validation import ensure_non_empty_frame


def test_ensure_non_empty_frame_accepts_non_empty_frames():
    ensure_non_empty_frame(pd.DataFrame({"value": [1]}), "boom")


def test_ensure_non_empty_frame_raises_for_empty_frames():
    with pytest.raises(ValueError, match="boom"):
        ensure_non_empty_frame(pd.DataFrame(), "boom")
