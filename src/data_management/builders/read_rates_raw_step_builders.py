import logging

import pandas as pd

from src.config.config import READ_RATES_RAW_DATA_STEP_DIR_PATH, config
from src.enums.data_enums.options_trade_underlying_ibex_database_enum import (
    OptionsTradeUnderlyingIbexDatabaseEnum,
)
from src.enums.data_enums.risk_free_rates_enum import RiskFreeRatesEnum

logger = logging.getLogger(__name__)


class RiskFreeRatesBuilder:
    OUTPUT_FILENAME = (
        READ_RATES_RAW_DATA_STEP_DIR_PATH
        / f"{config.data_config.read_rates_raw_config.output_filename}.csv"
    )

    @staticmethod
    def get_output_filename():
        return RiskFreeRatesBuilder.OUTPUT_FILENAME

    @staticmethod
    def unify_overnight_rate(
        risk_free_rates_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Create unified overnight rate column of NAs initially
        unified_overnight_rate_col_name = RiskFreeRatesEnum.OVERNIGHT_RATE.value
        risk_free_rates_df[unified_overnight_rate_col_name] = pd.NA

        # Fill unified overnight rate column with STR and with EONIA - 8.5 bps where STR is not available
        spread_str_eonia = config.data_config.read_rates_raw_config.spread_str_eonia
        cutoff_date_str_eonia = config.data_config.read_rates_raw_config.cutoff_date_str_eonia
        cutoff_date_eonia = pd.to_datetime(cutoff_date_str_eonia)
        eonia_rate_col_name = RiskFreeRatesEnum.EONIA_RATE.value
        str_rate_col_name = RiskFreeRatesEnum.STR_RATE.value

        mask_pre = risk_free_rates_df.index < cutoff_date_eonia
        risk_free_rates_df.loc[mask_pre, unified_overnight_rate_col_name] = risk_free_rates_df.loc[mask_pre, eonia_rate_col_name] - spread_str_eonia

        mask_post = risk_free_rates_df.index >= cutoff_date_eonia
        risk_free_rates_df.loc[mask_post, unified_overnight_rate_col_name] = risk_free_rates_df.loc[mask_post, str_rate_col_name]

        return risk_free_rates_df

    @staticmethod
    def reindex_to_options_trades_dates(
        risk_free_rates_df: pd.DataFrame,
        options_trade_underlying_ibex_df: pd.DataFrame
    ) -> pd.DataFrame:
        unique_dates = pd.to_datetime(options_trade_underlying_ibex_df[OptionsTradeUnderlyingIbexDatabaseEnum.SESSION_DATE.value].unique())
        unique_dates = pd.Series(unique_dates).sort_values().reset_index(drop=True)
        risk_free_rates_df = risk_free_rates_df.reindex(unique_dates)

        return risk_free_rates_df
    
    @staticmethod
    def build(
        rates_dfs_dict: dict[str, pd.DataFrame],
        options_trade_underlying_ibex_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Extract rates dataframes from dict
        dfs_list = list(rates_dfs_dict.values())
        
        # Outer join all dataframes
        risk_free_rates_df = dfs_list[0].join(dfs_list[1:], how="outer")

        # Unify overnight rate
        risk_free_rates_df = RiskFreeRatesBuilder.unify_overnight_rate(risk_free_rates_df)

        # Select only relevant columns (overnight + euribor)
        relevant_columns = config.data_config.read_rates_raw_config.free_risk_rates_columns
        risk_free_rates_df = risk_free_rates_df[relevant_columns].copy()

        # Forward fill to convert monthly Euribor into daily series (each day of the month should use the last published rate)
        euribor_columns = [
            RiskFreeRatesEnum.EURIBOR_3M_RATE.value,
            RiskFreeRatesEnum.EURIBOR_6M_RATE.value,
            RiskFreeRatesEnum.EURIBOR_12M_RATE.value,
        ]
        risk_free_rates_df[euribor_columns] = (
            risk_free_rates_df[euribor_columns].ffill()
        )

        # Reindex based on historical trade dates to avoid having dates in the risk free rates df that are not present in the trades df
        risk_free_rates_df = RiskFreeRatesBuilder.reindex_to_options_trades_dates(risk_free_rates_df, options_trade_underlying_ibex_df)

        # Save CSV
        risk_free_rates_df.to_csv(
            RiskFreeRatesBuilder.get_output_filename(),
            index=True,
            index_label=RiskFreeRatesEnum.DATE.value,
            encoding="utf-8",
            sep=";",
        )
        logger.info(
            f"RiskFreeRates (with shape {risk_free_rates_df.shape}) saved in: {RiskFreeRatesBuilder.get_output_filename()}."
        )

        return risk_free_rates_df
