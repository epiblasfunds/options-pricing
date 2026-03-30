import logging

import pandas as pd

from src.config.config import config
from src.data_management.builders import VolatilityBuilder
from src.data_management.loaders.read_raw_step_loader import ReadRawStepLoader
from src.data_management.loaders.underlying_step_loader import UnderlyingStepLoader
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.data_management.utils.data_type_utils import convert_data_types
from src.enums.data_enums import ContractTypeEnum, OptionTradesUnderlyingDBEnum
from src.exceptions.data_exceptions import (
    MissingValuesError,
    NegativeTradePriceError,
    TimeToExpirationOutOfRangeError,
    UnderlyingExecDatetimeAfterExecDatetimeError,
    UnderlyingExecDatetimeOutOfRangeError,
)

logger = logging.getLogger(__name__)


class VolatilityStepLoader:

    @staticmethod
    def read_step_databases() -> pd.DataFrame:
        output_file = VolatilityBuilder.get_output_filename()

        options_trade_volatility_ibex_df = pd.read_csv(
            output_file,
            sep=";",
            header=0,
            dtype="string",
        )

        options_trade_volatility_ibex_df = convert_data_types(
            df=options_trade_volatility_ibex_df,
            selected_columns_dict=config.data_config.volatility_config.volatility_db_columns,
            format_date="%Y-%m-%d",
            format_datetime="%Y-%m-%d %H:%M:%S.%f",
        )

        return options_trade_volatility_ibex_df

    @staticmethod
    def _clean_trades_wo_underlying(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean trades without underlying info
        """
        missing_underlying_info_mask = (
            df[OptionTradesUnderlyingDBEnum.FUTURE_CONTRACT_CODE].isna()
            | df[OptionTradesUnderlyingDBEnum.UNDERLYING_PRICE].isna()
            | df[OptionTradesUnderlyingDBEnum.UNDERLYING_EXEC_DATETIME].isna()
        )

        total_rows = len(df)
        n_dropped = int(missing_underlying_info_mask.sum())
        if n_dropped > 0:
            pct_dropped = (n_dropped / total_rows * 100) if total_rows > 0 else 0.0
            logger.info(
                "Dropping %s/%s option trades (%.2f%%) without future/underlying information.",
                n_dropped,
                total_rows,
                pct_dropped,
            )
            df = df.loc[~missing_underlying_info_mask].copy()

        return df

    @staticmethod
    def _filter_high_lag(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        lag_max = float(config.data_config.volatility_config.underlying_lag_max_minutes)

        lag_mask = df[OptionTradesUnderlyingDBEnum.UNDERLYING_LAG_MINUTES] > lag_max

        total_rows = len(df)
        n_dropped = int(lag_mask.sum())
        if n_dropped > 0:
            pct_dropped = (n_dropped / total_rows * 100) if total_rows > 0 else 0.0
            logger.info(
                "Dropping %s/%s option trades (%.2f%%) with underlying lag > %.0f minutes.",
                n_dropped,
                total_rows,
                pct_dropped,
                lag_max,
            )
            df = df.loc[~lag_mask].copy()

        return df

    @staticmethod
    def _clean_trades(
        df: pd.DataFrame,
    ):
        df = VolatilityStepLoader._clean_trades_wo_underlying(df=df)
        df = VolatilityStepLoader._clean_trades_wo_underlying(df=df)
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
    def load(force_reload=False):
        if force_reload or not VolatilityBuilder.get_output_filename().exists():
            # Read
            option_trades_underlying_df = UnderlyingStepLoader.load()
            _, _, rates_df = ReadRawStepLoader.load()

            # Clean and Filter trades with stale underlying (lag > threshold)
            option_trades_underlying_df = VolatilityStepLoader._clean_trades(
                option_trades_underlying_df
            )

            # Validate
            VolatilityStepLoader._validate_sources(
                option_trades_underlying_df, rates_df
            )

            # Build
            VolatilityBuilder.build(option_trades_underlying_df, rates_df)

        return VolatilityStepLoader.read_step_databases()
