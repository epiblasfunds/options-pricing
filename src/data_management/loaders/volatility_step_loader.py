import logging

import pandas as pd

from src.config.config import config
from src.data_management.builders import (
    OptionTradesUnderlyingBuilder,
    RatesBuilder,
    VolatilityBuilder,
)
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.data_management.utils.data_type_utils import convert_data_types
from src.enums.data_enums import ContractTypeEnum, OptionTradesUnderlyingDBEnum
from src.exceptions.data_exceptions import (
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
    TimeToExpirationOutOfRangeError,
    UnderlyingExecDatetimeAfterExecDatetimeError,
    UnderlyingExecDatetimeOutOfRangeError,
)

logger = logging.getLogger(__name__)


class VolatilityStepLoader:

    # READ
    @staticmethod
    def _read_option_trades_underlying_db() -> pd.DataFrame:
        df = pd.read_csv(
            OptionTradesUnderlyingBuilder.get_output_filename(),
            sep=";",
            header=0,
            dtype="string",
        )
        return df

    @staticmethod
    def _read_rates() -> pd.DataFrame:
        df = pd.read_csv(
            RatesBuilder.get_output_filename(),
            sep=";",
            header=0,
            dtype="string",
        )
        return df

    # VALIDATIONS
    @staticmethod
    def _validate_maturity(options_trade_underlying_ibex_df: pd.DataFrame):
        contract_code_col = OptionTradesUnderlyingDBEnum.OPTION_CONTRACT_CODE
        maturity_date_col = OptionTradesUnderlyingDBEnum.MATURITY_DATETIME
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
    def _validate_strike(options_trade_underlying_ibex_df: pd.DataFrame):

        strike_col = OptionTradesUnderlyingDBEnum.STRIKE_PRICE
        contract_code_col = OptionTradesUnderlyingDBEnum.OPTION_CONTRACT_CODE

        contract_code_series = options_trade_underlying_ibex_df[contract_code_col]
        strike_series = options_trade_underlying_ibex_df[strike_col]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_missings(df: pd.DataFrame):
        if df.isna().any().any():
            raise MissingValuesError(f"Missing values found for {df}.")
        else:
            return

    @staticmethod
    def _validate_underlying_exec_datetime_temporal_coherence(
        options_trade_underlying_ibex_df: pd.DataFrame,
    ):

        exec_datetime_col = OptionTradesUnderlyingDBEnum.EXEC_DATETIME
        underlying_exec_datetime_col = (
            OptionTradesUnderlyingDBEnum.UNDERLYING_EXEC_DATETIME
        )

        # Convert to datetime
        exec_datetime = pd.to_datetime(
            options_trade_underlying_ibex_df[exec_datetime_col], format="mixed"
        )
        underlying_exec_datetime = pd.to_datetime(
            options_trade_underlying_ibex_df[underlying_exec_datetime_col],
            format="mixed",
        )

        # Check temporal coherence
        mask = underlying_exec_datetime > exec_datetime
        if mask.any():
            sample = options_trade_underlying_ibex_df[mask].iloc[0]
            raise UnderlyingExecDatetimeAfterExecDatetimeError(
                f"UnderlyingExecDatetime occurs after ExecDatetime.\nExample: {sample}"
            )

    @staticmethod
    def _validate_underlying_exec_datetime_range(
        options_trade_underlying_ibex_df: pd.DataFrame,
    ):
        session_date_col = OptionTradesUnderlyingDBEnum.SESSION_DATE
        underlying_exec_datetime_col = (
            OptionTradesUnderlyingDBEnum.UNDERLYING_EXEC_DATETIME
        )

        # Convert to datetime
        session_date = pd.to_datetime(
            options_trade_underlying_ibex_df[session_date_col], format="mixed"
        )
        underlying_exec_datetime = pd.to_datetime(
            options_trade_underlying_ibex_df[underlying_exec_datetime_col],
            format="mixed",
        )

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
    def _validate_time_to_expiration(options_trade_underlying_ibex_df: pd.DataFrame):
        time_to_expiration_col = OptionTradesUnderlyingDBEnum.TIME_TO_EXPIRATION
        time_to_expiration = options_trade_underlying_ibex_df[
            time_to_expiration_col
        ].astype("float64")

        # Check non-negative
        if (time_to_expiration < 0).any():
            raise TimeToExpirationOutOfRangeError(
                "Time to expiration cannot be negative."
            )

    @staticmethod
    def _validate_options_trade_underlying_ibex_df(
        option_trades_underlying_df: pd.DataFrame,
    ):
        # Format of main columns
        for col in [
            OptionTradesUnderlyingDBEnum.TRADE_PRICE_OPTION,
            OptionTradesUnderlyingDBEnum.UNDERLYING_PRICE,
            OptionTradesUnderlyingDBEnum.STRIKE_PRICE,
            OptionTradesUnderlyingDBEnum.QUANTITY,
        ]:
            if (option_trades_underlying_df[col].astype("float64") <= 0.0).any():
                raise NegativeTradePriceError(
                    f"NegativeTradePriceError in column {col}"
                )

        # Validate: Missing, strike and maturity with contract code
        VolatilityStepLoader._validate_missings(option_trades_underlying_df)
        VolatilityStepLoader._validate_strike(option_trades_underlying_df)
        VolatilityStepLoader._validate_maturity(option_trades_underlying_df)

        # Validate underlying exec datetime coherence
        VolatilityStepLoader._validate_underlying_exec_datetime_temporal_coherence(
            option_trades_underlying_df
        )
        VolatilityStepLoader._validate_underlying_exec_datetime_range(
            option_trades_underlying_df
        )

        # Validate time to expiration and rate
        VolatilityStepLoader._validate_time_to_expiration(option_trades_underlying_df)

    @staticmethod
    def _validate_rates_df(rates_df: pd.DataFrame):
        # Validate missing values
        VolatilityStepLoader._validate_missings(rates_df)

    @staticmethod
    def _validate_sources(
        options_trade_underlying_ibex_df: pd.DataFrame, rates_df: pd.DataFrame
    ):
        VolatilityStepLoader._validate_options_trade_underlying_ibex_df(
            options_trade_underlying_ibex_df
        )
        VolatilityStepLoader._validate_rates_df(rates_df)

    @staticmethod
    def _convert_types(
        option_trades_underlying_df: pd.DataFrame, rates_df: pd.DataFrame
    ):
        option_trades_underlying_df = convert_data_types(
            df=option_trades_underlying_df,
            selected_columns_dict=config.data_config.underlying_config.option_trades_underlying_db_columns,
            format_date="%Y-%m-%d"
        )
        rates_df = convert_data_types(
            df=rates_df,
            selected_columns_dict=config.data_config.read_raw_config.rates_columns,
            format_date="%Y-%m-%d"
        )

        return option_trades_underlying_df, rates_df

    @staticmethod
    def load():
        # Read
        option_trades_underlying_df = (
            VolatilityStepLoader._read_option_trades_underlying_db()
        )
        rates_df = VolatilityStepLoader._read_rates()

        # Validate
        VolatilityStepLoader._validate_sources(option_trades_underlying_df, rates_df)

        # Conversion Type
        option_trades_underlying_df, rates_df = VolatilityStepLoader._convert_types(
            option_trades_underlying_df, rates_df
        )

        # Build
        options_trade_volatility_ibex_df = VolatilityBuilder.build(
            option_trades_underlying_df, rates_df
        )

        return options_trade_volatility_ibex_df
