import typing as t

import pandas as pd

from src.config.config import config
from src.data_management.read_raw_handler.abstract_read_raw_handler import (
    AbstractReadRawHandler,
)
from src.enums.data_enums.tgentrades_enum import TgentradesEnum
from src.exceptions.data_exceptions import DataError


class TgentradesHandler(AbstractReadRawHandler):
    def _custom_process(self, df: pd.DataFrame) -> pd.DataFrame:
        is_header = self._check_is_header(df.iloc[0])
        if is_header:
            skip_column_list = [
                c for c, v in df.iloc[0].items() if v.lower().strip() == "secuencia"
            ]
            df.drop(columns=skip_column_list, inplace=True)
            df = df.iloc[1:, :].reset_index(drop=True)

        return df

    def _filter_by_ibex(
        self, df: pd.DataFrame, contracts_prefixes: t.List[str]
    ) -> pd.DataFrame:

        df = df[
            df[TgentradesEnum.CONTRACT_CODE.value].str.startswith(
                tuple(contracts_prefixes), na=False
            )
        ]

        # Select only futures and options with monthly maturity
        # We can identify them because their number of characters
        df = df[
            df[TgentradesEnum.CONTRACT_CODE.value]
            .str.len()
            .isin(
                [
                    config.data_config.read_raw_config.n_characters_futures_code,
                    config.data_config.read_raw_config.n_characters_options_code,
                ]
            )
        ]

        return df

    def _validate(self) -> t.List[t.Tuple[DataError, str]]:
        pass
