import typing as t

import pandas as pd

from src.data_management.read_raw_handler.abstract_read_raw_handler import (
    AbstractReadRawHandler,
)
from src.exceptions.data_exceptions import DataError


class CContractsC2Handler(AbstractReadRawHandler):
    def _custom_process(self, df: pd.DataFrame) -> pd.DataFrame:
        is_header = self._check_is_header(df.iloc[0])
        if is_header:
            df = df.iloc[1:, :].reset_index(drop=True)

        return df

    def _validate(self) -> t.List[t.Tuple[DataError, str]]:
        # Validar que un mismo ContractCode tenga la misma maturity para distintos SessionDate



        pass
