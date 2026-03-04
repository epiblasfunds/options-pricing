import logging

import pandas as pd

from src.data_management.builders.read_rates_raw_step_builders import (
    RiskFreeRatesBuilder,
)
from src.data_management.builders.underlying_rates_step_builders import (
    OptionsTradeUnderlyingRatesIbexBuilder,
)
from src.data_management.builders.underlying_step_builders import (
    OptionsTradeUnderlyingIbexBuilder,
)
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.enums.data_enums.contract_type_enum import ContractTypeEnum
from src.enums.data_enums.options_trade_underlying_ibex_database_enum import (
    OptionsTradeUnderlyingIbexDatabaseEnum,
)
from src.enums.data_enums.risk_free_rates_enum import RiskFreeRatesEnum
from src.exceptions.data_exceptions import (
    DuplicatedPrimaryKeysError,
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
    RatesOutOfRangeError,
    UnderlyingExecDatetimeAfterExecDatetimeError,
    UnderlyingExecDatetimeOutOfRangeError,
)

logger = logging.getLogger(__name__)

class UnderlyingRatesStepLoader:

    # READ
    @staticmethod
    def _read_free_risk_rates_databases() -> pd.DataFrame:
        file_path = RiskFreeRatesBuilder.get_output_filename()
        df = pd.read_csv(
            file_path,
            sep=";",
            header=0,
            index_col=RiskFreeRatesEnum.DATE.value,
            parse_dates=True,
        )
        return df       

    @staticmethod
    def _read_options_trade_underlying_ibex_database() -> pd.DataFrame:
        file_path = OptionsTradeUnderlyingIbexBuilder.get_output_filename()
        df = pd.read_csv(
            file_path,
            sep=";",
            header=0,
            dtype="string",
        )
        return df
    
    @staticmethod
    def _read_free_risk_rates_and_options_trade_underlying_ibex_databases():
        free_risk_rates_df = UnderlyingRatesStepLoader._read_free_risk_rates_databases()
        options_trade_underlying_ibex_df = UnderlyingRatesStepLoader._read_options_trade_underlying_ibex_database()
        return free_risk_rates_df, options_trade_underlying_ibex_df
        
    # VALIDATIONS
    @staticmethod
    def _validate_maturity(
        options_trade_underlying_df: pd.DataFrame
    ):
        contract_code_col = OptionsTradeUnderlyingIbexDatabaseEnum.OPTION_CONTRACT_CODE.value
        maturity_date_col = OptionsTradeUnderlyingIbexDatabaseEnum.MATURITY_DATE.value
        session_date_col = OptionsTradeUnderlyingIbexDatabaseEnum.SESSION_DATE.value

        contract_code_series = options_trade_underlying_df[contract_code_col]
        maturity_series = options_trade_underlying_df[maturity_date_col]
        session_date_series = options_trade_underlying_df[session_date_col]

        validate_maturity_contract_code(
            contract_type=ContractTypeEnum.OPTIONS,
            contract_code_series=contract_code_series,
            maturity_series=maturity_series,
            session_date_series=session_date_series,
        )

    @staticmethod
    def _validate_strike(
        options_trade_underlying_df: pd.DataFrame
    ):

        strike_col = OptionsTradeUnderlyingIbexDatabaseEnum.STRIKE_PRICE.value
        contract_code_col = OptionsTradeUnderlyingIbexDatabaseEnum.OPTION_CONTRACT_CODE.value
        
        contract_code_series = options_trade_underlying_df[contract_code_col]
        strike_series = options_trade_underlying_df[strike_col]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_missings(
        options_trade_underlying_df: pd.DataFrame
    ):
        if options_trade_underlying_df.isna().any().any():
            raise MissingValuesError("Missing values found for options trade underlying ibex dataframe.")
        else:
            return

    @staticmethod
    def _validate_underlying_exec_datetime_temporal_coherence(
        options_trade_underlying_df: pd.DataFrame
    ):
        
        exec_datetime_col = OptionsTradeUnderlyingIbexDatabaseEnum.EXEC_DATETIME.value
        underlying_exec_datetime_col = OptionsTradeUnderlyingIbexDatabaseEnum.UNDERLYING_EXEC_DATETIME.value
        
        # Convert to datetime
        exec_datetime = pd.to_datetime(options_trade_underlying_df[exec_datetime_col])
        underlying_exec_datetime = pd.to_datetime(options_trade_underlying_df[underlying_exec_datetime_col])
        
        # Check temporal coherence
        mask = underlying_exec_datetime > exec_datetime
        if mask.any():
            sample = options_trade_underlying_df[mask].iloc[0]
            raise UnderlyingExecDatetimeAfterExecDatetimeError(
                f"UnderlyingExecDatetime occurs after ExecDatetime.\nExample: {sample}"
            )

    @staticmethod
    def _validate_underlying_exec_datetime_range(
        options_trade_underlying_df: pd.DataFrame
    ):
        session_date_col = OptionsTradeUnderlyingIbexDatabaseEnum.SESSION_DATE.value
        underlying_exec_datetime_col = OptionsTradeUnderlyingIbexDatabaseEnum.UNDERLYING_EXEC_DATETIME.value
        
        # Convert to datetime
        session_date = pd.to_datetime(options_trade_underlying_df[session_date_col])
        underlying_exec_datetime = pd.to_datetime(options_trade_underlying_df[underlying_exec_datetime_col])
        
        # Define session end (SessionDate 23:59:59)
        session_end = session_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Check that underlying_exec_datetime <= session_end
        # (we allow it to be before SessionDate for previous day trades)
        mask = underlying_exec_datetime > session_end
        if mask.any():
            sample = options_trade_underlying_df[mask].iloc[0]
            raise UnderlyingExecDatetimeOutOfRangeError(
                f"UnderlyingExecDatetime is after SessionDate end.\nExample: {sample}"
            )
    
    @staticmethod
    def _validate_options_trade_underlying_df(options_trade_underlying_df: pd.DataFrame):

        # Format of main columns
        trade_price_col = OptionsTradeUnderlyingIbexDatabaseEnum.TRADE_PRICE_OPTION.value
        quantity_col = OptionsTradeUnderlyingIbexDatabaseEnum.QUANTITY.value
        if (options_trade_underlying_df[trade_price_col].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (options_trade_underlying_df[quantity_col].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()

        # Validate: Missing, strike and maturity with contract code
        UnderlyingRatesStepLoader._validate_missings(options_trade_underlying_df)
        UnderlyingRatesStepLoader._validate_strike(options_trade_underlying_df)
        UnderlyingRatesStepLoader._validate_maturity(options_trade_underlying_df)

        # Validate underlying exec datetime coherence
        UnderlyingRatesStepLoader._validate_underlying_exec_datetime_temporal_coherence(options_trade_underlying_df)
        UnderlyingRatesStepLoader._validate_underlying_exec_datetime_range(options_trade_underlying_df)

    @staticmethod
    def _validate_free_risk_rates_df(free_risk_rates_df: pd.DataFrame):
        # Missing values
        if free_risk_rates_df.isna().any().any():
            raise MissingValuesError("Missing values found in risk-free rates dataframe.")
        
        # Duplicate dates in index
        if free_risk_rates_df.index.duplicated().any():
            raise DuplicatedPrimaryKeysError("Duplicate dates found in risk-free rates index.")
        
        # Check that all rate columns are numeric (already should be from reading)
        rate_columns = [col for col in free_risk_rates_df.columns]
        for col in rate_columns:
            if not pd.api.types.is_numeric_dtype(free_risk_rates_df[col]):
                raise ValueError(f"Column {col} is not numeric.")
        
        # Check that rates are in reasonable range (-2% to 15%)
        min_rate = -2.0
        max_rate = 15.0
        if (free_risk_rates_df < min_rate).any().any() or (free_risk_rates_df > max_rate).any().any():
            raise RatesOutOfRangeError(f"Interest rates must be between {min_rate}% and {max_rate}%.")

    @staticmethod
    def _validate_sources(free_risk_rates_df: pd.DataFrame, options_trade_underlying_ibex_df: pd.DataFrame):
        UnderlyingRatesStepLoader._validate_options_trade_underlying_df(options_trade_underlying_ibex_df)
        UnderlyingRatesStepLoader._validate_free_risk_rates_df(free_risk_rates_df)
    
    @staticmethod
    def load():
        free_risk_rates_df, options_trade_underlying_ibex_df =  UnderlyingRatesStepLoader._read_free_risk_rates_and_options_trade_underlying_ibex_databases()
        UnderlyingRatesStepLoader._validate_sources(free_risk_rates_df, options_trade_underlying_ibex_df)
        options_trade_underlying_rates_ibex_df = OptionsTradeUnderlyingRatesIbexBuilder.build(free_risk_rates_df, options_trade_underlying_ibex_df)
        return options_trade_underlying_rates_ibex_df