import logging
from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config
from src.enums.data_enums import (
    FuturesTradeIbexDatabaseEnum,
    OptionsTradeIbexDatabaseEnum,
)

logger = logging.getLogger(__name__)


def options_future_contract_relationship(
    options_trades_filename: Path,
    futures_trades_filename: Path,
) -> pd.DataFrame:

    # Read CSV options trades
    df_options = pd.read_csv(
        options_trades_filename,
        delimiter=";",
        header=0,
        dtype="string",
    )

    # Read CSV futures trades
    df_futures = pd.read_csv(
        futures_trades_filename,
        delimiter=";",
        header=0,
        dtype="string",
    )

    # Select relevant columns
    options_df = (
        df_options[
            [
                OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value,
                OptionsTradeIbexDatabaseEnum.MATURITY_DATE.value,
            ]
        ]
        .drop_duplicates(
            subset=[OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value],
            keep="first",
        )
        .copy()
    )

    futures_df = (
        df_futures[
            [
                FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE.value,
                FuturesTradeIbexDatabaseEnum.MATURITY_DATE.value,
            ]
        ]
        .drop_duplicates(
            subset=[FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE.value],
            keep="first",
        )
        .copy()
    )

    # Merge on MaturityDate
    df = options_df.merge(
        futures_df,
        how="left",
        on=OptionsTradeIbexDatabaseEnum.MATURITY_DATE.value,
    )

    # Save CSV
    PRODUCT_SPLIT_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    output_filename = (
        config.data_config.product_split_config.output_filename_relationship
    )
    output_file = PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    logger.info(f"DF (with shape {df.shape}) saved in: {output_file}.")

    return df
