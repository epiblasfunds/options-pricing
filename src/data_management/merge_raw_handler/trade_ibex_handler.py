import logging
import typing as t
from pathlib import Path

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH, config
from src.data_management.merge_raw_handler.get_contract_type_handler import (
    get_contract_type,
)
from src.enums.data_enums.trade_ibex_database_enum import TradeIbexDatabaseEnum

logger = logging.getLogger(__name__)


def merge_trade_with_contracts(
    trades_filename: str,
    contracts_filename: str,
    merge_columns: t.List[str],
    selected_columns_list: t.List[str],
) -> pd.DataFrame:

    # Read CSVs
    trades_df = pd.read_csv(
        Path(trades_filename),
        delimiter=";",
        header=0,
        dtype="string",
    )
    contracts_df = pd.read_csv(
        Path(contracts_filename),
        delimiter=";",
        header=0,
        dtype="string",
    )

    # Merge
    merged_df = trades_df.merge(
        contracts_df, on=merge_columns, how="left", suffixes=("", "_contract")
    )

    # Add type of contract
    merged_df[config.data_config.merge_raw_config.contract_type_column] = merged_df[
        TradeIbexDatabaseEnum.CONTRACT_CODE.value
    ].apply(get_contract_type)

    # Select only relevant columns
    merged_df = merged_df[
        selected_columns_list
        + [config.data_config.merge_raw_config.contract_type_column]
    ]

    # Save CSV
    MERGE_RAW_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    output_filename = config.data_config.merge_raw_config.output_filename
    output_file = MERGE_RAW_DATA_STEP_DIR_PATH / f"{output_filename}.csv"
    merged_df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    logger.info(f"DF (with shape {merged_df.shape}) saved in: {output_file}.")

    return merged_df
