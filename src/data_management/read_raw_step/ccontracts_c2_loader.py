import pandas as pd

from src.data_management.read_raw_step.abstract_read_raw_loader import (
    AbstractReadRawLoader,
)


class CContractsC2Loader(AbstractReadRawLoader):
    @classmethod
    def _custom_process(cls, df: pd.DataFrame) -> pd.DataFrame:
        is_header = cls._check_is_header(df.iloc[0])
        if is_header:
            df = df.iloc[1:, :].reset_index(drop=True)

        return df
