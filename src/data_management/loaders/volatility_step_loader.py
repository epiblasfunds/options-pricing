import logging

import pandas as pd

from src.data_management.builders.underlying_rates_step_builders import (
    OptionsTradeUnderlyingRatesIbexBuilder,
)
from src.data_management.builders.volatility_step_builders import (
    OptionsTradeVolatilityIbexBuilder,
)
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.enums.data_enums.contract_type_enum import ContractTypeEnum
from src.enums.data_enums.options_trade_underlying_rates_ibex_database_enum import (
    OptionsTradeUnderlyingRatesIbexDatabaseEnum,
)
from src.exceptions.data_exceptions import (
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
    RatesOutOfRangeError,
    TimeToMaturityOutOfRangeError,
    UnderlyingExecDatetimeAfterExecDatetimeError,
    UnderlyingExecDatetimeOutOfRangeError,
)

logger = logging.getLogger(__name__)


class VolatilityStepLoader:
 
    # READ
    @staticmethod
    def _read_options_trade_underlying_rates_ibex_database() -> pd.DataFrame:
        file_path = OptionsTradeUnderlyingRatesIbexBuilder.get_output_filename()
        df = pd.read_csv(
            file_path,
            sep=";",
            header=0,
            dtype="string",
        )
        return df

    
    # VALIDATIONS
    @staticmethod
    def _validate_maturity(
        options_trade_underlying_rates_df: pd.DataFrame
    ):
        contract_code_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.OPTION_CONTRACT_CODE.value
        maturity_date_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.MATURITY_DATE.value
        session_date_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.SESSION_DATE.value

        contract_code_series = options_trade_underlying_rates_df[contract_code_col]
        maturity_series = options_trade_underlying_rates_df[maturity_date_col]
        session_date_series = options_trade_underlying_rates_df[session_date_col]

        validate_maturity_contract_code(
            contract_type=ContractTypeEnum.OPTIONS,
            contract_code_series=contract_code_series,
            maturity_series=maturity_series,
            session_date_series=session_date_series,
        )

    @staticmethod
    def _validate_strike(
        options_trade_underlying_rates_df: pd.DataFrame
    ):

        strike_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.STRIKE_PRICE.value
        contract_code_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.OPTION_CONTRACT_CODE.value
        
        contract_code_series = options_trade_underlying_rates_df[contract_code_col]
        strike_series = options_trade_underlying_rates_df[strike_col]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_missings(
        options_trade_underlying_rates_df: pd.DataFrame
    ):
        if options_trade_underlying_rates_df.isna().any().any():
            raise MissingValuesError("Missing values found for options trade underlying ibex dataframe.")
        else:
            return

    @staticmethod
    def _validate_underlying_exec_datetime_temporal_coherence(
        options_trade_underlying_rates_df: pd.DataFrame
    ):
        
        exec_datetime_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.EXEC_DATETIME.value
        underlying_exec_datetime_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.UNDERLYING_EXEC_DATETIME.value
        
        # Convert to datetime
        exec_datetime = pd.to_datetime(options_trade_underlying_rates_df[exec_datetime_col])
        underlying_exec_datetime = pd.to_datetime(options_trade_underlying_rates_df[underlying_exec_datetime_col])
        
        # Check temporal coherence
        mask = underlying_exec_datetime > exec_datetime
        if mask.any():
            sample = options_trade_underlying_rates_df[mask].iloc[0]
            raise UnderlyingExecDatetimeAfterExecDatetimeError(
                f"UnderlyingExecDatetime occurs after ExecDatetime.\nExample: {sample}"
            )

    @staticmethod
    def _validate_underlying_exec_datetime_range(
        options_trade_underlying_rates_df: pd.DataFrame
    ):
        session_date_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.SESSION_DATE.value
        underlying_exec_datetime_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.UNDERLYING_EXEC_DATETIME.value
        
        # Convert to datetime
        session_date = pd.to_datetime(options_trade_underlying_rates_df[session_date_col])
        underlying_exec_datetime = pd.to_datetime(options_trade_underlying_rates_df[underlying_exec_datetime_col])
        
        # Define session end (SessionDate 23:59:59)
        session_end = session_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Check that underlying_exec_datetime <= session_end
        # (we allow it to be before SessionDate for previous day trades)
        mask = underlying_exec_datetime > session_end
        if mask.any():
            sample = options_trade_underlying_rates_df[mask].iloc[0]
            raise UnderlyingExecDatetimeOutOfRangeError(
                f"UnderlyingExecDatetime is after SessionDate end.\nExample: {sample}"
            )

    @staticmethod
    def _validate_time_to_maturity(
        options_trade_underlying_rates_df: pd.DataFrame
    ):
        time_to_maturity_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.TIME_TO_MATURITY.value
        time_to_maturity = options_trade_underlying_rates_df[time_to_maturity_col].astype("float64")
        
        # Check non-negative
        if (time_to_maturity < 0).any():
            raise TimeToMaturityOutOfRangeError("Time to maturity cannot be negative.")

    @staticmethod
    def _validate_risk_free_rate(
        options_trade_underlying_rates_df: pd.DataFrame
    ):
        risk_free_rate_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.RISK_FREE_RATE.value
        
        # Check numeric
        if not pd.api.types.is_numeric_dtype(options_trade_underlying_rates_df[risk_free_rate_col]):
            risk_free_rate = options_trade_underlying_rates_df[risk_free_rate_col].astype("float64")
        else:
            risk_free_rate = options_trade_underlying_rates_df[risk_free_rate_col]
        
        # Check reasonable range (-2% to 15%)
        min_rate = -2.0
        max_rate = 15.0
        if (risk_free_rate < min_rate).any() or (risk_free_rate > max_rate).any():
            raise RatesOutOfRangeError(f"Risk-free rate must be between {min_rate}% and {max_rate}%.")
        
    @staticmethod
    def _validate_options_trade_underlying_rates_df(options_trade_underlying_rates_ibex_df: pd.DataFrame):
        # Format of main columns
        trade_price_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.TRADE_PRICE_OPTION.value
        quantity_col = OptionsTradeUnderlyingRatesIbexDatabaseEnum.QUANTITY.value
        if (options_trade_underlying_rates_ibex_df[trade_price_col].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (options_trade_underlying_rates_ibex_df[quantity_col].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()

        # Validate: Missing, strike and maturity with contract code
        VolatilityStepLoader._validate_missings(options_trade_underlying_rates_ibex_df)
        VolatilityStepLoader._validate_strike(options_trade_underlying_rates_ibex_df)
        VolatilityStepLoader._validate_maturity(options_trade_underlying_rates_ibex_df)

        # Validate underlying exec datetime coherence
        VolatilityStepLoader._validate_underlying_exec_datetime_temporal_coherence(options_trade_underlying_rates_ibex_df)
        VolatilityStepLoader._validate_underlying_exec_datetime_range(options_trade_underlying_rates_ibex_df)
        
        # Validate time to maturity and risk-free rate
        VolatilityStepLoader._validate_time_to_maturity(options_trade_underlying_rates_ibex_df)
        VolatilityStepLoader._validate_risk_free_rate(options_trade_underlying_rates_ibex_df)
    
    @staticmethod
    def load():
        options_trade_underlying_rates_ibex_df = VolatilityStepLoader._read_options_trade_underlying_rates_ibex_database()
        VolatilityStepLoader._validate_options_trade_underlying_rates_df(options_trade_underlying_rates_ibex_df)
        options_trade_volatility_ibex_df = OptionsTradeVolatilityIbexBuilder.build(options_trade_underlying_rates_ibex_df)
        return options_trade_volatility_ibex_df
