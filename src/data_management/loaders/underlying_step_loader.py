import logging
import typing as t

import pandas as pd

from src.config.config import UNDERLYING_DATA_STEP_DIR_PATH, config
from src.data_management.builders import (
    FuturesTradeIbexBuilder,
    OptionsTradeIbexBuilder,
    OptionsUnderlyingIbexBuilder,
)
from src.data_management.builders.underlying_step_builders import (
    OptionsTradeUnderlyingIbexBuilder,
)
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.enums.data_enums import CcontractsC2Enum, ContractTypeEnum, TgentradesEnum
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
    def _validate_maturity(contracts_df: pd.DataFrame, contract_type: ContractTypeEnum):
        contract_code_series = contracts_df[CcontractsC2Enum.CONTRACT_CODE.value]
        maturity_series = contracts_df[CcontractsC2Enum.MATURITY_DATE.value]
        session_date_series = contracts_df[CcontractsC2Enum.SESSION_DATE.value]

        validate_maturity_contract_code(
            contract_type=contract_type,
            contract_code_series=contract_code_series,
            maturity_series=maturity_series,
            session_date_series=session_date_series,
        )

    @staticmethod
    def _validate_strike(contracts_df: pd.DataFrame):
        contract_code_series = contracts_df[CcontractsC2Enum.CONTRACT_CODE.value]
        strike_series = contracts_df[CcontractsC2Enum.STRIKE_PRICE.value]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_missing_ccontracts(contracts_df: pd.DataFrame):
        cc_series = contracts_df[CcontractsC2Enum.CONTRACT_CODE.value]

        options_contracts_mask = (
            cc_series.str.len()
            == config.data_config.contract_code_config.options_code_len
        )
        options = contracts_df[options_contracts_mask]

        futures_contracts_mask = (
            cc_series.str.len()
            == config.data_config.contract_code_config.futures_code_len
        )
        future_columns = [
            c for c in contracts_df.columns if c != CcontractsC2Enum.STRIKE_PRICE.value
        ]
        futures = contracts_df[futures_contracts_mask][future_columns]

        if options.isna().any().any() or futures.isna().any().any():
            raise MissingValuesError()

    @staticmethod
    def _validate_trades_df(trades_df):
        # Format validations
        if (trades_df["TradePrice"].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (trades_df["Quantity"].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()

        # Unique Primary Keys
        MergeRawStepLoader._validate_primary_keys(
            df=trades_df,
            pk_columns=[
                TgentradesEnum.TRADE_EXEC_ID.value,
            ],
        )

        # NAs
        if trades_df.isna().any().any():
            raise MissingValuesError()

    @staticmethod
    def _validate_options(contracts_df):
        # Unique Primary Keys
        MergeRawStepLoader._validate_primary_keys(
            df=contracts_df,
            pk_columns=[
                CcontractsC2Enum.SESSION_DATE.value,
                CcontractsC2Enum.CONTRACT_CODE.value,
            ],
        )

        # Validate maturity with contract code
        MergeRawStepLoader._validate_maturity(contracts_df, ContractTypeEnum.OPTIONS)
        MergeRawStepLoader._validate_maturity(contracts_df, ContractTypeEnum.FUTURES)

        # Validate strikes with contract code
        MergeRawStepLoader._validate_strike(contracts_df)

        # NAs
        MergeRawStepLoader._validate_missing_ccontracts(contracts_df)

    @staticmethod
    def _validate_sources(
        options_trade_ibex_df: pd.DataFrame,
        futures_trade_ibex_df: pd.DataFrame,
        options_underlying_ibex_df: pd.DataFrame,
    ):
        UnderlyingStepLoader._validate_options(options_trade_ibex_df)
        UnderlyingStepLoader._validate_futures(futures_trade_ibex_df)
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
