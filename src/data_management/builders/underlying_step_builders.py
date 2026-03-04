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

    @staticmethod
    def create_exec_datetime(
        df: pd.DataFrame, exec_time_col: str, session_date_col: str
    ) -> pd.Series:
        # Extract only the time component in case the column contains a full datetime
        df[exec_time_col] = df[exec_time_col].astype(str).str.split().str[-1]

        # Ensure microseconds are present by appending ".000000" when missing
        df[exec_time_col] = df[exec_time_col].apply(
            lambda x: x if "." in x else x + ".000000"
        )

        # Combine date and time and convert to pandas datetime
        return pd.to_datetime(
            df[session_date_col].astype(str) + " " + df[exec_time_col],
            format="%Y-%m-%d %H:%M:%S.%f",
        )

    @classmethod
    def build(
        cls,
        options_df: pd.DataFrame,
        futures_df: pd.DataFrame,
        options_underlying_ibex_df: pd.DataFrame,
    ) -> pd.DataFrame:

        # ---------------------------------------------------------------------
        # Pre‑filter underlying lookup table
        # Some option-to-future relationships may lack a maturity or future code;
        # dropping them now prevents carrying invalid rows forward (which would
        # later manifest as NaNs when we merge or perform the as-of join).
        options_underlying_ibex_df = options_underlying_ibex_df.dropna(
            subset=[OptionsUnderlyingIbexDatabaseEnum.MATURITY_DATE.value]
        )

        # Keep only option trades for which we have a (remaining) underlying
        valid_option_codes = (
            options_underlying_ibex_df[OptionsUnderlyingIbexDatabaseEnum.OPTION_CONTRACT_CODE.value]
            .unique()
        )
        mask_options_with_underlying = (
            options_df[OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value]
            .isin(valid_option_codes)
        )
        options_df = options_df[mask_options_with_underlying]

        # drop option trades where maturity is missing – they cannot be matched to
        # an underlying future contract later and only introduce NaNs
        options_df = options_df.dropna(subset=[OptionsTradeIbexDatabaseEnum.MATURITY_DATE.value])

        # Create exec_datetime (SessionDate + ExecTime)
        exec_datetime_col = OptionsTradeUnderlyingIbexDatabaseEnum.EXEC_DATETIME.value
        underlying_exec_datetime_col = (
            OptionsTradeUnderlyingIbexDatabaseEnum.UNDERLYING_EXEC_DATETIME.value
        )
        options_df[exec_datetime_col] = cls.create_exec_datetime(
            df=options_df,
            exec_time_col=OptionsTradeIbexDatabaseEnum.EXEC_TIME.value,
            session_date_col=OptionsTradeIbexDatabaseEnum.SESSION_DATE.value,
        )
        futures_df[underlying_exec_datetime_col] = cls.create_exec_datetime(
            df=futures_df,
            exec_time_col=FuturesTradeIbexDatabaseEnum.EXEC_TIME.value,
            session_date_col=FuturesTradeIbexDatabaseEnum.SESSION_DATE.value,
        )

        # Join option with its underlying future
        options_trade_ibex_df = options_df.merge(
            options_underlying_ibex_df,
            on=OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value,
            how="left",
        )

        # Order by exec_datetime
        options_trade_ibex_df = options_trade_ibex_df.sort_values(
            exec_datetime_col
        ).reset_index(drop=True)
        futures_trade_ibex_df = futures_df.sort_values(
            underlying_exec_datetime_col
        ).reset_index(drop=True)

        # As-of join: Last trade of the underlying FUTURE with exec_datetime <= exec_datetime of the option
        df = pd.merge_asof(
            options_trade_ibex_df,
            futures_trade_ibex_df,
            by=FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE.value,
            left_on=exec_datetime_col,
            right_on=underlying_exec_datetime_col,
            direction="backward",
            suffixes=("", "_future"),
        )

        # Rename columns
        trade_price_option = (
            OptionsTradeUnderlyingIbexDatabaseEnum.TRADE_PRICE_OPTION.value
        )
        underlying_price = OptionsTradeUnderlyingIbexDatabaseEnum.UNDERLYING_PRICE.value
        df = df.rename(
            columns={
                OptionsTradeIbexDatabaseEnum.TRADE_PRICE.value: trade_price_option,
                f"{FuturesTradeIbexDatabaseEnum.TRADE_PRICE.value}_future": underlying_price,
            }
        )

        # remove rows where we failed to match a past future trade
        df = df.dropna(subset=[underlying_exec_datetime_col])

        df = df[
            config.data_config.underlying_config.options_trade_underlying_ibex_database_columns
        ]

        # Save CSV
        output_file = cls.get_output_filename()
        df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

        logger.info(
            f"OptionsTradeUnderlyingIbexDatabase (with shape {df.shape}) saved in: {output_file}."
        )

        return df
