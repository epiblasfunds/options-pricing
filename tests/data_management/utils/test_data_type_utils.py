from datetime import date

import pandas as pd
import pytest

from src.data_management.utils.data_type_utils import convert_data_types
from src.data_management.utils.data_type_utils import prepare_time
from src.enums.data_enums import DataTypeEnum
from src.exceptions.data_exceptions import DataTypeConversionError


def test_prepare_time_normalizes_supported_formats():
    assert prepare_time("01/01/3000 9:00:06") == "09:00:06.000000"
    assert prepare_time("09:00:00.000000") == "09:00:00.000000"
    assert prepare_time("09:00:00:000") == "09:00:00.000"
    assert prepare_time("08:00:16.002360") == "08:00:16.002360"


def test_prepare_time_rejects_unexpected_format():
    with pytest.raises(ValueError):
        prepare_time("09")


def test_convert_data_types_converts_date_time_and_numeric_columns():
    frame = pd.DataFrame(
        {
            "date_col": ["20260131"],
            "datetime_col": ["2026-01-31 09:30:00"],
            "time_col": ["9:30:00"],
            "float_col": ["1,25"],
            "int_col": ["7"],
        }
    )

    converted = convert_data_types(
        frame,
        {
            "date_col": DataTypeEnum.DATE.value,
            "datetime_col": DataTypeEnum.DATETIME.value,
            "time_col": DataTypeEnum.TIME.value,
            "float_col": DataTypeEnum.FLOAT.value,
            "int_col": DataTypeEnum.INT.value,
        },
    )

    assert converted.loc[0, "date_col"] == date(2026, 1, 31)
    assert str(converted.loc[0, "datetime_col"]) == "2026-01-31 09:30:00"
    assert converted.loc[0, "time_col"].hour == 9
    assert converted.loc[0, "float_col"] == pytest.approx(1.25)
    assert converted.loc[0, "int_col"] == 7


def test_convert_data_types_wraps_conversion_errors():
    with pytest.raises(DataTypeConversionError):
        convert_data_types(
            pd.DataFrame({"date_col": ["not-a-date"]}),
            {"date_col": DataTypeEnum.DATE.value},
        )
