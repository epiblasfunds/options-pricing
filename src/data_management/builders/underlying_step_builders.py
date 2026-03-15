import logging
from pathlib import Path

import pandas as pd

from src.config.config import UNDERLYING_DATA_STEP_DIR_PATH, config
from src.enums.data_enums import (
    FuturesTradeIbexDBEnum,
    OptionsTradeIbexDBEnum,
    OptionTradesUnderlyingDBEnum,
    OptionUnderlyingDBEnum,
)

logger = logging.getLogger(__name__)


class OptionTradesUnderlyingBuilder:
    @staticmethod
    def get_output_filename() -> Path:
        return (
            UNDERLYING_DATA_STEP_DIR_PATH
            / f"{config.data_config.underlying_config.output_filename}.csv"
        )

    @staticmethod
    def _to_csv(df: pd.DataFrame):
        df = df.copy()

        # Format Datetimes
        for col in [
            OptionTradesUnderlyingDBEnum.EXEC_DATETIME,
            OptionTradesUnderlyingDBEnum.UNDERLYING_EXEC_DATETIME,
            # OptionTradesUnderlyingDBEnum.MATURITY_DATETIME,
        ]:
            df[col] = df[col].dt.strftime(date_format="%Y-%m-%d %H:%M:%S.%f")

        df.to_csv(
            OptionTradesUnderlyingBuilder.get_output_filename(),
            index=False,
            encoding="utf-8",
            sep=";",
        )
        logger.info(
            f"OptionsTradeUnderlyingIbexDatabase (with shape {df.shape}) "
            + f"saved in: {OptionTradesUnderlyingBuilder.get_output_filename()}."
        )

    @classmethod
    def build(
        cls,
        options_df: pd.DataFrame,
        futures_df: pd.DataFrame,
        options_underlying_df: pd.DataFrame,
    ) -> pd.DataFrame:

        # Keep only option trades for which we have a underlying
        valid_option_codes = options_underlying_df[
            OptionUnderlyingDBEnum.OPTION_CONTRACT_CODE
        ].unique()
        mask_options_with_underlying = options_df[
            OptionsTradeIbexDBEnum.OPTION_CONTRACT_CODE
        ].isin(valid_option_codes)
        options_df = options_df[mask_options_with_underlying]

        # Join option with its underlying future
        option_trades_df = options_df.merge(
            options_underlying_df,
            on=OptionsTradeIbexDBEnum.OPTION_CONTRACT_CODE,
            how="left",
        )

        # Convert EXEC_DATETIME to datetime for merge_asof
        option_trades_df[OptionsTradeIbexDBEnum.EXEC_DATETIME] = pd.to_datetime(
            option_trades_df[OptionsTradeIbexDBEnum.EXEC_DATETIME], format="%Y-%m-%d %H:%M:%S.%f"
        )
        future_trades_df = futures_df.copy()
        future_trades_df[FuturesTradeIbexDBEnum.EXEC_DATETIME] = pd.to_datetime(
            future_trades_df[FuturesTradeIbexDBEnum.EXEC_DATETIME], format="%Y-%m-%d %H:%M:%S.%f"
        )

        # Rename EXEC_DATETIME in futures to UNDERLYING_EXEC_DATETIME for maintaining both in the merged df
        future_trades_df[OptionTradesUnderlyingDBEnum.UNDERLYING_EXEC_DATETIME] = (
            future_trades_df[FuturesTradeIbexDBEnum.EXEC_DATETIME]
        )

        # Order by exec_datetime
        option_trades_df = option_trades_df.sort_values(
            OptionsTradeIbexDBEnum.EXEC_DATETIME
        ).reset_index(drop=True)
        future_trades_df = future_trades_df.sort_values(
            FuturesTradeIbexDBEnum.EXEC_DATETIME
        ).reset_index(drop=True)

        # As-of join: Last trade of the underlying FUTURE with exec_datetime <= exec_datetime of the option
        df = pd.merge_asof(
            option_trades_df,
            future_trades_df,
            by=FuturesTradeIbexDBEnum.FUTURE_CONTRACT_CODE,
            left_on=OptionsTradeIbexDBEnum.EXEC_DATETIME,
            right_on=OptionTradesUnderlyingDBEnum.UNDERLYING_EXEC_DATETIME,
            direction="backward",
            suffixes=("", "_future"),
        )

        # Rename columns
        renamed_columns = {
            OptionsTradeIbexDBEnum.TRADE_PRICE: OptionTradesUnderlyingDBEnum.TRADE_PRICE_OPTION,
            f"{FuturesTradeIbexDBEnum.TRADE_PRICE}_future": OptionTradesUnderlyingDBEnum.UNDERLYING_PRICE,
        }
        df = df.rename(columns=renamed_columns)

        columns = list(
            config.data_config.underlying_config.option_trades_underlying_db_columns.keys()
        )
        df = df[columns]

        # Delete rows with missing underlying price (i.e. no underlying trade found before option trade)
        df = df.dropna(subset=[OptionTradesUnderlyingDBEnum.UNDERLYING_PRICE])

        # Save CSV
        OptionTradesUnderlyingBuilder._to_csv(df)

        return df
