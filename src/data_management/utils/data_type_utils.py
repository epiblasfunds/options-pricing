import typing as t
from enum import Enum

import pandas as pd

from src.enums.data_enums import DataTypeEnum
from src.exceptions.data_exceptions import DataTypeConversionError


# Process datetimes
def prepare_time(time_str: str) -> str:
    """
    Possible values for time_str: [
        "01/01/3000 9:00:06",
        "01/01/3000 09:00:06",
        "09:00:00.000000",
        "09:00:00:000",
        "08:00:16.002360"
    ]
    """
    time_str = time_str.split(" ")[-1]
    time_str_split = time_str.split(":")
    time_str_split[0] = (
        f"0{time_str_split[0]}" if len(time_str_split[0]) == 1 else time_str_split[0]
    )

    if len(time_str_split) == 4:
        time_str = f"{time_str_split[0]}:{time_str_split[1]}:{time_str_split[2]}.{time_str_split[3]}"
    elif len(time_str_split) != 3:
        raise ValueError(f"Unexpected time format: {time_str}")
    else:
        time_str = ":".join(time_str_split)

    time_str_split = time_str.split(".")
    if len(time_str_split) == 1:
        time_str = f"{time_str_split[0]}.000000"

    return time_str


# Convert data types
def convert_data_types(
    df: pd.DataFrame,
    selected_columns_dict: t.Dict[Enum, DataTypeEnum],
    format_date="%Y%m%d",
    format_datetime="%Y-%m-%d %H:%M:%S",
    format_time="%H:%M:%S.%f"
) -> pd.DataFrame:
    for col, dtype in selected_columns_dict.items():
        try:
            if dtype == DataTypeEnum.DATE.value:
                df[col] = pd.to_datetime(df[col], format=format_date).dt.date
            elif dtype == DataTypeEnum.DATETIME:
                df[col] = pd.to_datetime(df[col], format=format_datetime)
            elif dtype == DataTypeEnum.TIME:
                df[col] = df[col].str.strip().str.split(" ").str[-1].apply(prepare_time)
                df[col] = pd.to_datetime(df[col], format=format_time)
            elif dtype == DataTypeEnum.FLOAT.value:
                df[col] = pd.to_numeric(df[col].str.replace(",", "."), downcast="float")
            elif dtype == DataTypeEnum.INT.value:
                df[col] = pd.to_numeric(df[col], downcast="integer")
        except Exception as e:
            raise DataTypeConversionError(
                f"DataTypeConversionError: {col} cannot be converted to {dtype}"
            ) from e

    return df
