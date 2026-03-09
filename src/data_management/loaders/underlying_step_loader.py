import logging
import typing as t

import pandas as pd

from src.config.config import UNDERLYING_DATA_STEP_DIR_PATH, config
from src.data_management.builders import (
    FuturesTradeIbexBuilder,
    OptionsTradeIbexBuilder,
    OptionsTradeUnderlyingIbexBuilder,
    OptionsUnderlyingIbexBuilder,
)
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.enums.data_enums import (
    CcontractsC2Enum,
    ContractTypeEnum,
    FuturesTradeIbexDatabaseEnum,
    OptionsTradeIbexDatabaseEnum,
)
from src.enums.data_enums.options_underlying_ibex_database_enum import (
    OptionsUnderlyingIbexDatabaseEnum,
)
from src.exceptions.data_exceptions import (
    DuplicatedPrimaryKeysError,
    MissingValuesError,
    NegativeQuantityError,
    NegativeTimeToExpirationError,
    NegativeTradePriceError,
    SessionAfterMaturityError,
)

logger = logging.getLogger(__name__)


class UnderlyingStepLoader:
    OUTPUT_FILENAME = (
        UNDERLYING_DATA_STEP_DIR_PATH
        / f"{config.data_config.underlying_config.output_filename}.csv"
    )

    # READ
    @staticmethod
    def _read_source_dfs() -> None:
        options_trade_ibex_df = pd.read_csv(
            OptionsTradeIbexBuilder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )
        futures_trade_ibex_df = pd.read_csv(
            FuturesTradeIbexBuilder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )
        options_underlying_ibex_df = pd.read_csv(
            OptionsUnderlyingIbexBuilder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )

        return options_trade_ibex_df, futures_trade_ibex_df, options_underlying_ibex_df

    # VALIDATIONS
    @staticmethod
    def _validate_maturity(
        trade_ibex_df: pd.DataFrame, contract_type: ContractTypeEnum
    ):
        if contract_type == ContractTypeEnum.OPTIONS:
            contract_code_col = OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE
            maturity_date_col = OptionsTradeIbexDatabaseEnum.MATURITY_DATE
            session_date_col = OptionsTradeIbexDatabaseEnum.SESSION_DATE
        else:
            contract_code_col = FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE
            maturity_date_col = FuturesTradeIbexDatabaseEnum.MATURITY_DATE
            session_date_col = FuturesTradeIbexDatabaseEnum.SESSION_DATE

        contract_code_series = trade_ibex_df[contract_code_col]
        maturity_series = trade_ibex_df[maturity_date_col]
        session_date_series = trade_ibex_df[session_date_col]

        validate_maturity_contract_code(
            contract_type=contract_type,
            contract_code_series=contract_code_series,
            maturity_series=maturity_series,
            session_date_series=session_date_series,
        )

    @staticmethod
    def _validate_strike(trade_ibex_df: pd.DataFrame, contract_type: ContractTypeEnum):
        if contract_type == ContractTypeEnum.FUTURES:
            return

        contract_code_col = OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE
        strike_col = OptionsTradeIbexDatabaseEnum.STRIKE_PRICE

        contract_code_series = trade_ibex_df[contract_code_col]
        strike_series = trade_ibex_df[strike_col]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_missings(
        trade_ibex_df: pd.DataFrame, contract_type: ContractTypeEnum
    ):
        if contract_type == ContractTypeEnum.FUTURES:
            selected_cols = [
                c
                for c in trade_ibex_df.columns
                if c != FuturesTradeIbexDatabaseEnum.STRIKE_PRICE
            ]
            trade_ibex_df = trade_ibex_df[selected_cols]

        if trade_ibex_df.isna().any().any():
            raise MissingValuesError("Missing values found for Options.")
        else:
            return

    @staticmethod
    def _validate_trade_df(
        trade_ibex_df: pd.DataFrame, contract_type: ContractTypeEnum
    ):
        if contract_type == ContractTypeEnum.OPTIONS:
            maturity_col = OptionsTradeIbexDatabaseEnum.MATURITY_DATE
            session_date_col = OptionsTradeIbexDatabaseEnum.SESSION_DATE
            trade_price_col = OptionsTradeIbexDatabaseEnum.TRADE_PRICE
            quantity_col = OptionsTradeIbexDatabaseEnum.QUANTITY
            time_to_expiration_col = OptionsTradeIbexDatabaseEnum.TIME_TO_EXPIRATION
        else:
            maturity_col = FuturesTradeIbexDatabaseEnum.MATURITY_DATE
            session_date_col = FuturesTradeIbexDatabaseEnum.SESSION_DATE
            trade_price_col = FuturesTradeIbexDatabaseEnum.TRADE_PRICE
            quantity_col = FuturesTradeIbexDatabaseEnum.QUANTITY
            time_to_expiration_col = FuturesTradeIbexDatabaseEnum.TIME_TO_EXPIRATION

        # Format validations
        if (trade_ibex_df[trade_price_col].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (trade_ibex_df[quantity_col].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()
        
        if (trade_ibex_df[time_to_expiration_col].astype("float64") < 0.0).any():
            raise NegativeTimeToExpirationError()

        # Validate maturity with contract code
        UnderlyingStepLoader._validate_maturity(trade_ibex_df, contract_type)

        # Validate maturity and session date coherence
        session = pd.to_datetime(trade_ibex_df[session_date_col])
        maturity = pd.to_datetime(trade_ibex_df[maturity_col])

        mask = session > maturity
        if mask.any():
            sample = trade_ibex_df[mask].iloc[0]
            raise SessionAfterMaturityError(f"SessionDate occurs after MaturityDate.\nExample: {sample}.")

        # Validate strikes with contract code
        UnderlyingStepLoader._validate_strike(trade_ibex_df, contract_type)
        
        # NAs
        UnderlyingStepLoader._validate_missings(trade_ibex_df, contract_type)

    @staticmethod
    def _validate_underlying_candidates(options_underlying_ibex_df: pd.DataFrame):
        # Duplicates
        if options_underlying_ibex_df.duplicated(
            subset=[
                OptionsUnderlyingIbexDatabaseEnum.OPTION_CONTRACT_CODE,
                OptionsUnderlyingIbexDatabaseEnum.MATURITY_DATE,
                OptionsUnderlyingIbexDatabaseEnum.FUTURE_CONTRACT_CODE,
            ]
        ).any():
            raise DuplicatedPrimaryKeysError(
                "Duplicated primary keys found for Options-Underlying candidates."
            )

        # NAs
        if options_underlying_ibex_df.isna().any().any():
            raise MissingValuesError(
                "Missing values found for Options-Underlying candidates."
            )

    @staticmethod
    def _validate_sources(
        options_trade_ibex_df: pd.DataFrame,
        futures_trade_ibex_df: pd.DataFrame,
        options_underlying_ibex_df: pd.DataFrame,
    ):
        UnderlyingStepLoader._validate_trade_df(
            options_trade_ibex_df, ContractTypeEnum.OPTIONS
        )
        UnderlyingStepLoader._validate_trade_df(
            futures_trade_ibex_df, ContractTypeEnum.FUTURES
        )
        UnderlyingStepLoader._validate_underlying_candidates(options_underlying_ibex_df)

    @staticmethod
    def load():
        options_trade_ibex_df, futures_trade_ibex_df, options_underlying_ibex_df = (
            UnderlyingStepLoader._read_source_dfs()
        )
        UnderlyingStepLoader._validate_sources(
            options_trade_ibex_df, futures_trade_ibex_df, options_underlying_ibex_df
        )
        options_trade_underlying_ibex_db = OptionsTradeUnderlyingIbexBuilder.build(
            options_trade_ibex_df, futures_trade_ibex_df, options_underlying_ibex_df
        )
        return options_trade_underlying_ibex_db
