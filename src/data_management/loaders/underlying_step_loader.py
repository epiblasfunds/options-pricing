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
from src.exceptions.data_exceptions import (
    DuplicatedPrimaryKeysError,
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
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
    def _validate_primary_keys(df: pd.DataFrame, pk_columns: t.List[str]):
        pk_df = df[pk_columns]
        dup_mask = pk_df.duplicated()
        if dup_mask.any():
            first_dup = pk_df[dup_mask].iloc[0]
            raise DuplicatedPrimaryKeysError(
                "MergeRawHandler::_validate_contracts_df. Duplicate (SessionDate, ContractCode) pair found: "
                f"SessionDate={first_dup[CcontractsC2Enum.SESSION_DATE.value]}, "
                f"ContractCode={first_dup[CcontractsC2Enum.CONTRACT_CODE.value]}."
            )

    @staticmethod
    def _validate_maturity(
        trade_ibex_df: pd.DataFrame, contract_type: ContractTypeEnum
    ):
        if contract_type == ContractTypeEnum.OPTIONS:
            contract_code_col = OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value
            maturity_date_col = OptionsTradeIbexDatabaseEnum.MATURITY_DATE.value
            session_date_col = OptionsTradeIbexDatabaseEnum.SESSION_DATE.value
        else:
            contract_code_col = FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE.value
            maturity_date_col = FuturesTradeIbexDatabaseEnum.MATURITY_DATE.value
            session_date_col = FuturesTradeIbexDatabaseEnum.SESSION_DATE.value

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

        contract_code_col = OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value
        strike_col = OptionsTradeIbexDatabaseEnum.STRIKE_PRICE.value

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
                if c != FuturesTradeIbexDatabaseEnum.STRIKE_PRICE.value
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
            trade_exec_id_col = OptionsTradeIbexDatabaseEnum.TRADE_EXEC_ID.value
            trade_price_col = OptionsTradeIbexDatabaseEnum.TRADE_PRICE.value
            quantity_col = OptionsTradeIbexDatabaseEnum.QUANTITY.value
        else:
            trade_exec_id_col = FuturesTradeIbexDatabaseEnum.TRADE_EXEC_ID.value
            trade_price_col = FuturesTradeIbexDatabaseEnum.TRADE_PRICE.value
            quantity_col = FuturesTradeIbexDatabaseEnum.QUANTITY.value

        # Format validations
        if (trade_ibex_df[trade_price_col].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (trade_ibex_df[quantity_col].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()

        # Unique Primary Keys
        UnderlyingStepLoader._validate_primary_keys(
            df=trade_ibex_df,
            pk_columns=[
                trade_exec_id_col,
            ],
        )

        # Validate maturity with contract code
        UnderlyingStepLoader._validate_maturity(trade_ibex_df, contract_type)

        # Validate strikes with contract code
        UnderlyingStepLoader._validate_strike(trade_ibex_df, contract_type)

        # NAs
        UnderlyingStepLoader._validate_missings(trade_ibex_df, contract_type)

    @staticmethod
    def _validate_underlying_candidates(options_underlying_ibex_df: pd.DataFrame):
        # Unique Primary Keys
        UnderlyingStepLoader._validate_primary_keys(
            df=options_underlying_ibex_df,
            pk_columns=[
                OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value,
                FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE.value,
            ],
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
