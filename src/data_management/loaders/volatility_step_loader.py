import logging

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH
from src.data_management.builders import (
    OptionTradesUnderlyingBuilder,
    RatesBuilder,
    VolatilityBuilder,
)
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.enums.data_enums import ContractTypeEnum, OptionTradesUnderlyingDBEnum
from src.exceptions.data_exceptions import (
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
    RatesOutOfRangeError,
    TimeToExpirationOutOfRangeError,
    UnderlyingExecDatetimeAfterExecDatetimeError,
    UnderlyingExecDatetimeOutOfRangeError,
)

logger = logging.getLogger(__name__)


class VolatilityStepLoader:
 
    # READ
    @staticmethod
    def _read_options_trade_underlying_ibex_database() -> pd.DataFrame:
        file_path = OptionTradesUnderlyingBuilder.get_output_filename()
        df = pd.read_csv(
            file_path,
            sep=";",
            header=0,
            dtype="string",
        )
        return df
    
    @staticmethod
    def _read_rates() -> pd.DataFrame:
        df = pd.read_csv(
            MERGE_RAW_DATA_STEP_DIR_PATH / RatesBuilder.get_output_filename(),
            sep=";",
            header=0,
            dtype="string",
        )
        return df

    
    # VALIDATIONS
    @staticmethod
    def _validate_maturity(
        options_trade_underlying_ibex_df: pd.DataFrame
    ):
        contract_code_col = OptionTradesUnderlyingDBEnum.OPTION_CONTRACT_CODE
        maturity_date_col = OptionTradesUnderlyingDBEnum.MATURITY_DATE
        session_date_col = OptionTradesUnderlyingDBEnum.SESSION_DATE

        contract_code_series = options_trade_underlying_ibex_df[contract_code_col]
        maturity_series = options_trade_underlying_ibex_df[maturity_date_col]
        session_date_series = options_trade_underlying_ibex_df[session_date_col]

        validate_maturity_contract_code(
            contract_type=ContractTypeEnum.OPTIONS,
            contract_code_series=contract_code_series,
            maturity_series=maturity_series,
            session_date_series=session_date_series,
        )

    @staticmethod
    def _validate_strike(
        options_trade_underlying_ibex_df: pd.DataFrame
    ):

        strike_col = OptionTradesUnderlyingDBEnum.STRIKE_PRICE
        contract_code_col = OptionTradesUnderlyingDBEnum.OPTION_CONTRACT_CODE
        
        contract_code_series = options_trade_underlying_ibex_df[contract_code_col]
        strike_series = options_trade_underlying_ibex_df[strike_col]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_missings(
        df: pd.DataFrame
    ):
        if df.isna().any().any():
            raise MissingValuesError(f"Missing values found for {df}.")
        else:
            return

    @staticmethod
    def _validate_underlying_exec_datetime_temporal_coherence(
        options_trade_underlying_ibex_df: pd.DataFrame
    ):
        
        exec_datetime_col = OptionTradesUnderlyingDBEnum.EXEC_DATETIME
        underlying_exec_datetime_col = OptionTradesUnderlyingDBEnum.UNDERLYING_EXEC_DATETIME
        
        # Convert to datetime
        exec_datetime = pd.to_datetime(options_trade_underlying_ibex_df[exec_datetime_col], format='mixed')
        underlying_exec_datetime = pd.to_datetime(options_trade_underlying_ibex_df[underlying_exec_datetime_col], format='mixed')
        
        # Check temporal coherence
        mask = underlying_exec_datetime > exec_datetime
        if mask.any():
            sample = options_trade_underlying_ibex_df[mask].iloc[0]
            raise UnderlyingExecDatetimeAfterExecDatetimeError(
                f"UnderlyingExecDatetime occurs after ExecDatetime.\nExample: {sample}"
            )

    @staticmethod
    def _validate_underlying_exec_datetime_range(
        options_trade_underlying_ibex_df: pd.DataFrame
    ):
        session_date_col = OptionTradesUnderlyingDBEnum.SESSION_DATE
        underlying_exec_datetime_col = OptionTradesUnderlyingDBEnum.UNDERLYING_EXEC_DATETIME
        
        # Convert to datetime
        session_date = pd.to_datetime(options_trade_underlying_ibex_df[session_date_col], format='mixed')
        underlying_exec_datetime = pd.to_datetime(options_trade_underlying_ibex_df[underlying_exec_datetime_col], format='mixed')
        
        # Define session end (SessionDate 23:59:59)
        session_end = session_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Check that underlying_exec_datetime <= session_end
        # (we allow it to be before SessionDate for previous day trades)
        mask = underlying_exec_datetime > session_end
        if mask.any():
            sample = options_trade_underlying_ibex_df[mask].iloc[0]
            raise UnderlyingExecDatetimeOutOfRangeError(
                f"UnderlyingExecDatetime is after SessionDate end.\nExample: {sample}"
            )

    @staticmethod
    def _validate_time_to_expiration(
        options_trade_underlying_ibex_df: pd.DataFrame
    ):
        time_to_expiration_col = OptionTradesUnderlyingDBEnum.TIME_TO_EXPIRATION
        time_to_expiration = options_trade_underlying_ibex_df[time_to_expiration_col].astype("float64")
        
        # Check non-negative
        if (time_to_expiration < 0).any():
            raise TimeToExpirationOutOfRangeError("Time to expiration cannot be negative.")
        
    @staticmethod
    def _validate_options_trade_underlying_ibex_df(options_trade_underlying_ibex_df: pd.DataFrame):
        # Format of main columns
        trade_price_col = OptionTradesUnderlyingDBEnum.TRADE_PRICE_OPTION
        quantity_col = OptionTradesUnderlyingDBEnum.QUANTITY
        if (options_trade_underlying_ibex_df[trade_price_col].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (options_trade_underlying_ibex_df[quantity_col].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()

        # Validate: Missing, strike and maturity with contract code
        VolatilityStepLoader._validate_missings(options_trade_underlying_ibex_df)
        VolatilityStepLoader._validate_strike(options_trade_underlying_ibex_df)
        VolatilityStepLoader._validate_maturity(options_trade_underlying_ibex_df)

        # Validate underlying exec datetime coherence
        VolatilityStepLoader._validate_underlying_exec_datetime_temporal_coherence(options_trade_underlying_ibex_df)
        VolatilityStepLoader._validate_underlying_exec_datetime_range(options_trade_underlying_ibex_df)
        
        # Validate time to expiration and rate
        VolatilityStepLoader._validate_time_to_expiration(options_trade_underlying_ibex_df)

    @staticmethod
    def _validate_rates_df(rates_df: pd.DataFrame):
        # Validate missing values
        VolatilityStepLoader._validate_missings(rates_df)
    
    @staticmethod
    def _validate_sources(
        options_trade_underlying_ibex_df: pd.DataFrame,
        rates_df: pd.DataFrame
    ):
        VolatilityStepLoader._validate_options_trade_underlying_ibex_df(options_trade_underlying_ibex_df)
        VolatilityStepLoader._validate_rates_df(rates_df)
    
    @staticmethod
    def load():
        options_trade_underlying_ibex_df = VolatilityStepLoader._read_options_trade_underlying_ibex_database()
        rates_df = VolatilityStepLoader._read_rates()
        VolatilityStepLoader._validate_sources(options_trade_underlying_ibex_df, rates_df)
        options_trade_volatility_ibex_df = VolatilityBuilder.build(options_trade_underlying_ibex_df, rates_df)
        return options_trade_volatility_ibex_df
