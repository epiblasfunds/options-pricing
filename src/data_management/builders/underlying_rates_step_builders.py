import logging

import numpy as np
import pandas as pd

from src.config.config import UNDERLYING_RATES_DATA_STEP_DIR_PATH, config
from src.enums.data_enums.options_trade_underlying_ibex_database_enum import (
    OptionsTradeUnderlyingIbexDatabaseEnum,
)
from src.enums.data_enums.options_trade_underlying_rates_ibex_database_enum import (
    OptionsTradeUnderlyingRatesIbexDatabaseEnum,
)
from src.enums.data_enums.risk_free_rates_enum import RiskFreeRatesEnum

logger = logging.getLogger(__name__)

class OptionsTradeUnderlyingRatesIbexBuilder:
    OUTPUT_FILENAME = (
        UNDERLYING_RATES_DATA_STEP_DIR_PATH
        / f"{config.data_config.underlying_rates_config.output_filename}.csv"
    )

    @staticmethod
    def get_output_filename():
        return OptionsTradeUnderlyingRatesIbexBuilder.OUTPUT_FILENAME
    
    @staticmethod
    def create_time_to_maturity_column(
        options_trade_underlying_ibex_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        # Convert time columns to datetime
        exec_datetime_col = OptionsTradeUnderlyingIbexDatabaseEnum.EXEC_DATETIME.value
        maturity_date_col = OptionsTradeUnderlyingIbexDatabaseEnum.MATURITY_DATE.value
        options_trade_underlying_ibex_df[exec_datetime_col] = pd.to_datetime(options_trade_underlying_ibex_df[exec_datetime_col], format='ISO8601')
        options_trade_underlying_ibex_df[maturity_date_col] = pd.to_datetime(options_trade_underlying_ibex_df[maturity_date_col], format="%Y-%m-%d")

        # Calculate time to maturity in days with decimals
        time_to_maturity_col_name = OptionsTradeUnderlyingRatesIbexDatabaseEnum.TIME_TO_MATURITY.value
        options_trade_underlying_ibex_df.loc[:,time_to_maturity_col_name] = options_trade_underlying_ibex_df.loc[:,maturity_date_col] - options_trade_underlying_ibex_df.loc[:,exec_datetime_col]
        options_trade_underlying_ibex_df[time_to_maturity_col_name] = options_trade_underlying_ibex_df[time_to_maturity_col_name].dt.total_seconds() / (24*3600)

        return options_trade_underlying_ibex_df
    
    @staticmethod
    def calculate_discount_factors(
        risk_free_rates_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[int]]:
        """
        Convert risk free rates to discount factors for each rate and each day. The discount factor is calculated as:
        DF = exp(-r * T / 360), where r is in % (e.g., -0.35685%) and T is the tenor in days (e.g., 3, 6, 12 months -> 90, 180, 360 days).  
        """
        
        # Calculate discount factor for each rate each day
        tenors_days = config.data_config.underlying_rates_config.tenors_days
        discount_factors_df = pd.DataFrame(index=risk_free_rates_df.index)
        
        # Get rate columns (exclude 'Date' column)
        rate_columns = [col for col in risk_free_rates_df.columns if col != RiskFreeRatesEnum.DATE.value]
        
        for tenor, col in zip(tenors_days, rate_columns):
            # In decimals
            discount_factors_df[f'DF_{tenor}d'] = np.exp(-risk_free_rates_df[col].values / 100 * tenor / 360)
        
        return discount_factors_df, tenors_days
    
    @staticmethod
    def interpolate_risk_free_rates(
        options_trade_underlying_rates_ibex_df: pd.DataFrame,
        discount_factors_df: pd.DataFrame,
        tenors_days: list[int]
    ) -> pd.DataFrame:
        """
        Interpolate discount factors for each trade based on time to maturity and convert back to risk free rates.
        The interpolation is done in log space to ensure that the discount factors remain positive after interpolation.
        The risk free rate is calculated as:
        r = -ln(DF) * 360 / T * 100, where DF is the interpolated discount factor and T is the time to maturity in days.
        """

        # Get execution dates and time to maturity for each trade
        exec_dates = options_trade_underlying_rates_ibex_df[OptionsTradeUnderlyingRatesIbexDatabaseEnum.EXEC_DATETIME.value].dt.date
        discount_factors_df.index = discount_factors_df.index.date
        ttm_days = options_trade_underlying_rates_ibex_df[OptionsTradeUnderlyingRatesIbexDatabaseEnum.TIME_TO_MATURITY.value]

        # Obtain dfs values for each trade date
        dfs_by_trade = discount_factors_df.loc[exec_dates].values
        ln_dfs_by_trade = np.log(dfs_by_trade)

        # Interpolate ln(dfs) for each trade based on time to maturity
        interpolated_ln_dfs = np.empty(len(ttm_days))
        for i, ttm in enumerate(ttm_days):
            interpolated_ln_dfs[i] = np.interp(ttm, tenors_days, ln_dfs_by_trade[i], 
                                        left=ln_dfs_by_trade[i, 0], right=ln_dfs_by_trade[i, -1])
        
        # Convert back to rates
        interpolated_dfs = np.exp(interpolated_ln_dfs)

        # Recover risk free rate from interpolated discount factor
        risk_free_rate_values = -np.log(interpolated_dfs) * 360 / ttm_days * 100

        # Create new column with interpolated risk free rate
        options_trade_underlying_rates_ibex_df.loc[:,OptionsTradeUnderlyingRatesIbexDatabaseEnum.RISK_FREE_RATE.value] = risk_free_rate_values
        
        return options_trade_underlying_rates_ibex_df
        
    @staticmethod
    def calculate_risk_free_rate(
        options_trade_underlying_rates_ibex_df: pd.DataFrame,
        risk_free_rates_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        # Convert rates to float
        Overnight_rate_col = RiskFreeRatesEnum.OVERNIGHT_RATE.value
        Euribor_3M_rate_col = RiskFreeRatesEnum.EURIBOR_3M_RATE.value
        Euribor_6M_rate_col = RiskFreeRatesEnum.EURIBOR_6M_RATE.value
        Euribor_12M_rate_col = RiskFreeRatesEnum.EURIBOR_12M_RATE.value
        rates_cols = [Overnight_rate_col, Euribor_3M_rate_col, Euribor_6M_rate_col, Euribor_12M_rate_col]
        for col in rates_cols:
            risk_free_rates_df[col] = risk_free_rates_df[col].astype(float)

        # Obtain discount factor for each rate each day
        discount_factors_df, tenors_days = OptionsTradeUnderlyingRatesIbexBuilder.calculate_discount_factors(risk_free_rates_df)
        
        # Interpolate discount factor for each trade based on time to maturity
        options_trade_underlying_rates_ibex_df = OptionsTradeUnderlyingRatesIbexBuilder.interpolate_risk_free_rates(options_trade_underlying_rates_ibex_df, discount_factors_df, tenors_days)
    
        return options_trade_underlying_rates_ibex_df
    
    @staticmethod
    def build(
        risk_free_rates_df: pd.DataFrame,
        options_trade_underlying_ibex_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        # Calculate time to maturity for each trade
        options_trade_underlying_rates_ibex_df = OptionsTradeUnderlyingRatesIbexBuilder.create_time_to_maturity_column(options_trade_underlying_ibex_df)

        # Calculate free risk rate for each trade
        options_trade_underlying_rates_ibex_df = OptionsTradeUnderlyingRatesIbexBuilder.calculate_risk_free_rate(options_trade_underlying_rates_ibex_df, risk_free_rates_df)

        # Save CSV
        options_trade_underlying_rates_ibex_df.to_csv(
            OptionsTradeUnderlyingRatesIbexBuilder.get_output_filename(),
            encoding="utf-8",
            sep=";",
        )
        logger.info(
            f"OptionsTradeUnderlyingRatesIbex (with shape {options_trade_underlying_rates_ibex_df.shape}) saved in: {OptionsTradeUnderlyingRatesIbexBuilder.get_output_filename()}."
        )

        return options_trade_underlying_rates_ibex_df
        
    


















