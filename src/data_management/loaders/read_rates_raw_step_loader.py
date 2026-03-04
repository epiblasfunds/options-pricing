import logging

import pandas as pd

from src.config.config import (
    READ_RATES_RAW_DATA_STEP_DIR_PATH,
    RISK_FREE_RATES_DATA_DIR_PATH,
    config,
)
from src.data_management.builders.read_rates_raw_step_builders import (
    RiskFreeRatesBuilder,
)
from src.data_management.builders.underlying_step_builders import (
    OptionsTradeUnderlyingIbexBuilder,
)
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.enums.data_enums.contract_type_enum import ContractTypeEnum
from src.enums.data_enums.futures_trade_ibex_database_enum import (
    FuturesTradeIbexDatabaseEnum,
)
from src.enums.data_enums.options_trade_underlying_ibex_database_enum import (
    OptionsTradeUnderlyingIbexDatabaseEnum,
)
from src.enums.data_enums.risk_free_rates_enum import RiskFreeRatesEnum
from src.exceptions.data_exceptions import (
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
    UnderlyingExecDatetimeAfterExecDatetimeError,
    UnderlyingExecDatetimeOutOfRangeError,
)

logger = logging.getLogger(__name__)


class ReadRatesRawStepLoader:

    # READ
    @staticmethod
    def _load_rates_raw_csv(
        file_prefix: str,
    ) -> pd.DataFrame:
        
        # Search for files with the given prefix 
        matching_file = list(RISK_FREE_RATES_DATA_DIR_PATH.glob(f"*{file_prefix}*.csv"))
               
        # Read CSV
        file_path = matching_file[0]
        df = pd.read_csv(
            file_path,
            delimiter=",",
            header=0,
            dtype="string",
        )
        
        return df
    
    @staticmethod
    def _clean_rates_df(
        df: pd.DataFrame,
        rate_name: str,
    ) -> pd.DataFrame:
        
        # Select columns
        idx_date_col = config.data_config.read_rates_raw_config.idx_date_column
        idx_rate_col = config.data_config.read_rates_raw_config.idx_rate_column

        df = df.iloc[:, [idx_date_col, idx_rate_col]]

        # Map rate_name to enum value
        rate_enum_mapping = {
            "EONIA": RiskFreeRatesEnum.EONIA_RATE.value,
            "STR": RiskFreeRatesEnum.STR_RATE.value,
            "EURIBOR3M": RiskFreeRatesEnum.EURIBOR_3M_RATE.value,
            "EURIBOR6M": RiskFreeRatesEnum.EURIBOR_6M_RATE.value,
            "EURIBOR12M": RiskFreeRatesEnum.EURIBOR_12M_RATE.value,
        }
        
        # Rename column using enum value
        rate_col_name = rate_enum_mapping.get(rate_name, f"{rate_name}RATE")
        df.columns.values[idx_rate_col] = rate_col_name
        
        # Convert rate column to float
        df[rate_col_name] = pd.to_numeric(df[rate_col_name])

        # Convert date column to datetime and set as index
        index_rates_col_name = config.data_config.read_rates_raw_config.index_rates_column_name
        df[index_rates_col_name] = pd.to_datetime(df[index_rates_col_name])
        df.set_index(index_rates_col_name, inplace=True)

        return df
    
    @staticmethod
    def _read_rates_raw_databases() -> dict[str, pd.DataFrame]:
        cfg = config.data_config.read_rates_raw_config
        prefixes = cfg.get_rates_prefixes()

        rates = {}

        for rate_name, prefix in prefixes.items():
            raw_df = ReadRatesRawStepLoader._load_rates_raw_csv(prefix)
            clean_df = ReadRatesRawStepLoader._clean_rates_df(raw_df, rate_name)
            rates[rate_name] = clean_df

        return rates

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
        ReadRatesRawStepLoader._validate_missings(options_trade_underlying_df)
        ReadRatesRawStepLoader._validate_strike(options_trade_underlying_df)
        ReadRatesRawStepLoader._validate_maturity(options_trade_underlying_df)

        # Validate underlying exec datetime coherence
        ReadRatesRawStepLoader._validate_underlying_exec_datetime_temporal_coherence(options_trade_underlying_df)
        ReadRatesRawStepLoader._validate_underlying_exec_datetime_range(options_trade_underlying_df)
    
    @staticmethod
    def load():
        rates_dfs_dict = ReadRatesRawStepLoader._read_rates_raw_databases()
        options_trade_underlying_ibex_df = ReadRatesRawStepLoader._read_options_trade_underlying_ibex_database()
        ReadRatesRawStepLoader._validate_options_trade_underlying_df(options_trade_underlying_ibex_df)
        risk_free_rates_df = RiskFreeRatesBuilder.build(rates_dfs_dict, options_trade_underlying_ibex_df)
        return risk_free_rates_df


    

    

    
    
    
