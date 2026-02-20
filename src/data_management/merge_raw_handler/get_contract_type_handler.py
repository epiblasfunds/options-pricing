import pandas as pd

from src.config.config import config
from src.enums.contract_type_enum import ContractTypeEnum as contract_type_enum


def get_contract_type(
        code: str
) -> str:

    if pd.isna(code):
        return pd.NA

    if len(code) == config.data_config.read_raw_config.n_characters_options_code:
        return contract_type_enum.OPTIONS.value

    elif len(code) == config.data_config.read_raw_config.n_characters_futures_code:
        return contract_type_enum.FUTURES.value

    else:
        return pd.NA