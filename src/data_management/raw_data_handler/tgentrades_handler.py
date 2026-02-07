import typing as t

import pandas as pd

from src.data_management.raw_data_handler.abstract_raw_data_handler import (
    AbstractRawDataHandler,
)
from src.exceptions.data_exceptions import DataError


class TgentradesHandler(AbstractRawDataHandler):
    def _custom_process(self, df: pd.DataFrame) -> pd.DataFrame:
        is_header = self._check_is_header(df.iloc[0])
        if is_header:
            skip_column_list = [
                c for c, v in df.iloc[0].items() if v.lower().strip() == "secuencia"
            ]
            df.drop(columns=skip_column_list, inplace=True)
            df = df.iloc[1:, :].reset_index(drop=True)

        return df

    def _validate(self) -> t.List[t.Tuple[DataError, str]]:
        pass
