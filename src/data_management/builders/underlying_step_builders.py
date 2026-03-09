import logging
from pathlib import Path

import pandas as pd

from src.config.config import UNDERLYING_DATA_STEP_DIR_PATH, config
from src.enums.data_enums import (
    FuturesTradeIbexDatabaseEnum,
    OptionsTradeIbexDatabaseEnum,
    OptionsTradeUnderlyingIbexDatabaseEnum,
    OptionsUnderlyingIbexDatabaseEnum,
)

logger = logging.getLogger(__name__)


class OptionsTradeUnderlyingIbexBuilder:
    @staticmethod
    def get_output_filename() -> Path:
        return (
            UNDERLYING_DATA_STEP_DIR_PATH
            / f"{config.data_config.underlying_config.output_filename}.csv"
        )

    @classmethod
    def build(
        cls,
        options_df: pd.DataFrame,
        futures_df: pd.DataFrame,
        options_underlying_ibex_df: pd.DataFrame,
    ) -> pd.DataFrame:

        # Keep only option trades for which we have a underlying
        valid_option_codes = (
            options_underlying_ibex_df[OptionsUnderlyingIbexDatabaseEnum.OPTION_CONTRACT_CODE]
            .unique()
        )
        mask_options_with_underlying = (
            options_df[OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE]
            .isin(valid_option_codes)
        )
        options_df = options_df[mask_options_with_underlying]

        # Join option with its underlying future
        options_trade_ibex_df = options_df.merge(
            options_underlying_ibex_df,
            on=OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE,
            how="left",
        )

        # Convert EXEC_DATETIME to datetime for merge_asof
        options_trade_ibex_df[OptionsTradeIbexDatabaseEnum.EXEC_DATETIME] = pd.to_datetime(
            options_trade_ibex_df[OptionsTradeIbexDatabaseEnum.EXEC_DATETIME], format='mixed')
        futures_trade_ibex_df = futures_df.copy()
        futures_trade_ibex_df[FuturesTradeIbexDatabaseEnum.EXEC_DATETIME] = pd.to_datetime(
            futures_trade_ibex_df[FuturesTradeIbexDatabaseEnum.EXEC_DATETIME], format='mixed')
        
        # Rename EXEC_DATETIME in futures to UNDERLYING_EXEC_DATETIME for maintaining both in the merged df
        futures_trade_ibex_df[OptionsTradeUnderlyingIbexDatabaseEnum.UNDERLYING_EXEC_DATETIME] = (
            futures_trade_ibex_df[FuturesTradeIbexDatabaseEnum.EXEC_DATETIME]
        )

        # Order by exec_datetime
        options_trade_ibex_df = options_trade_ibex_df.sort_values(
            OptionsTradeIbexDatabaseEnum.EXEC_DATETIME
        ).reset_index(drop=True)
        futures_trade_ibex_df = futures_trade_ibex_df.sort_values(
            FuturesTradeIbexDatabaseEnum.EXEC_DATETIME
        ).reset_index(drop=True)

        # As-of join: Last trade of the underlying FUTURE with exec_datetime <= exec_datetime of the option
        df = pd.merge_asof(
            options_trade_ibex_df,
            futures_trade_ibex_df,
            by=FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE,
            left_on=OptionsTradeIbexDatabaseEnum.EXEC_DATETIME,
            right_on=OptionsTradeUnderlyingIbexDatabaseEnum.UNDERLYING_EXEC_DATETIME,
            direction="backward",
            suffixes=("", "_future"),
        )

        # Rename columns
        df = df.rename(
            columns={
                OptionsTradeIbexDatabaseEnum.TRADE_PRICE: OptionsTradeUnderlyingIbexDatabaseEnum.TRADE_PRICE_OPTION,
                f"{FuturesTradeIbexDatabaseEnum.TRADE_PRICE}_future": OptionsTradeUnderlyingIbexDatabaseEnum.UNDERLYING_PRICE,
            }
        )

        df = df[
            config.data_config.underlying_config.options_trade_underlying_ibex_database_columns
        ]

        # Delete rows with missing underlying price (i.e. no underlying trade found before option trade)
        df = df.dropna(subset=[OptionsTradeUnderlyingIbexDatabaseEnum.UNDERLYING_PRICE])

        # Save CSV
        output_file = cls.get_output_filename()
        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8",
            sep=";"
        )

        logger.info(
            f"OptionsTradeUnderlyingIbexDatabase (with shape {df.shape}) saved in: {output_file}."
        )

        return df
