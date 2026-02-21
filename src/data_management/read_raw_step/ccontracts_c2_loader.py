import typing as t

import pandas as pd

from src.config.config import config
from src.data_management.read_raw_step.abstract_read_raw_loader import (
    AbstractReadRawLoader,
)
from src.enums.data_enums.ccontracts_c2_enum import CcontractsC2Enum
from src.exceptions.data_exceptions import DataError


class CContractsC2Handler(AbstractReadRawLoader):
    def _custom_process(self, df: pd.DataFrame) -> pd.DataFrame:
        is_header = self._check_is_header(df.iloc[0])
        if is_header:
            df = df.iloc[1:, :].reset_index(drop=True)

        return df

    def _filter_by_ibex(
        self, df: pd.DataFrame, contracts_prefixes: t.List[str]
    ) -> pd.DataFrame:

        df = df[
            df[CcontractsC2Enum.CONTRACT_CODE.value].str.startswith(
                tuple(contracts_prefixes), na=False
            )
        ]

        # Select only futures and options with monthly maturity
        # We can identify them because their number of characters
        df = df[
            df[CcontractsC2Enum.CONTRACT_CODE.value]
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
        # Validar que un mismo ContractCode tenga la misma maturity para distintos SessionDate

        pass
