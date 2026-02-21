import pandas as pd

from src.data_management.read_raw_step.abstract_read_raw_loader import (
    AbstractReadRawLoader,
)


class CContractsC2Loader(AbstractReadRawLoader):
    def _custom_process(self, df: pd.DataFrame) -> pd.DataFrame:
        is_header = self._check_is_header(df.iloc[0])
        if is_header:
            df = df.iloc[1:, :].reset_index(drop=True)

        return df
