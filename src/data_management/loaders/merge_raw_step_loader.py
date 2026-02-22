import logging
import typing as t

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH, config
from src.data_management.builders import (
    CContractsC2Builder,
    TgentradesBuilder,
    TradeIbexBuilder,
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


class MergeRawStepLoader:
    OUTPUT_FILENAME = (
        MERGE_RAW_DATA_STEP_DIR_PATH
        / f"{config.data_config.merge_raw_config.output_filename}.csv"
    )

    # READ
    @staticmethod
    def _read_trades_and_contracts_dfs() -> None:
        trades_df = pd.read_csv(
            TgentradesBuilder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )
        contracts_df = pd.read_csv(
            CContractsC2Builder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )

        return trades_df, contracts_df

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
    def _validate_contracts_df(contracts_df):
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
    def _validate_sources(trades_df: pd.DataFrame, contracts_df: pd.DataFrame):
        MergeRawStepLoader._validate_trades_df(trades_df)
        MergeRawStepLoader._validate_contracts_df(contracts_df)

    @staticmethod
    def load():
        trades_df, contracts_df = MergeRawStepLoader._read_trades_and_contracts_dfs()
        MergeRawStepLoader._validate_sources(trades_df, contracts_df)
        trade_ibex_db = TradeIbexBuilder.build(trades_df, contracts_df)
        return trade_ibex_db
