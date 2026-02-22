import logging
import typing as t
from pathlib import Path

import pandas as pd

from src.config.config import (
    MERGE_RAW_DATA_STEP_DIR_PATH,
    RAW_DATA_STEP_DIR_PATH,
    config,
)
from src.data_management.utils import (
    get_contract_type,
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.enums.data_enums.ccontracts_c2_enum import CcontractsC2Enum
from src.enums.data_enums.contract_type_enum import ContractTypeEnum
from src.enums.data_enums.tgentrades_enum import TgentradesEnum
from src.enums.data_enums.trade_ibex_database_enum import TradeIbexDatabaseEnum
from src.exceptions.data_exceptions import (
    DuplicatedPrimaryKeysError,
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
)

logger = logging.getLogger(__name__)


class TradeIbexLoader:
    # READ
    @staticmethod
    def _read_trades_and_contracts_dfs() -> None:
        trades_filename = (
            RAW_DATA_STEP_DIR_PATH
            / f"{config.data_config.read_raw_config.tgentrades_prefix}.csv"
        )
        trades_df = pd.read_csv(
            Path(trades_filename),
            delimiter=";",
            header=0,
            dtype="string",
        )
        contracts_filename = (
            RAW_DATA_STEP_DIR_PATH
            / f"{config.data_config.read_raw_config.cconctracts_c2_prefix}.csv"
        )
        contracts_df = pd.read_csv(
            Path(contracts_filename),
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

        options_contracts_mask = cc_series.str.len() == config.data_config.contract_code_config.options_code_len
        options = contracts_df[options_contracts_mask]

        futures_contracts_mask = cc_series.str.len() == config.data_config.contract_code_config.futures_code_len
        future_columns = [c for c in contracts_df.columns if c != CcontractsC2Enum.STRIKE_PRICE.value]
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
        TradeIbexLoader._validate_primary_keys(
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
        TradeIbexLoader._validate_primary_keys(
            df=contracts_df,
            pk_columns=[
                CcontractsC2Enum.SESSION_DATE.value,
                CcontractsC2Enum.CONTRACT_CODE.value,
            ],
        )

        # Validate maturity with contract code
        TradeIbexLoader._validate_maturity(contracts_df, ContractTypeEnum.OPTIONS)
        TradeIbexLoader._validate_maturity(contracts_df, ContractTypeEnum.FUTURES)

        # Validate strikes with contract code
        TradeIbexLoader._validate_strike(contracts_df)

        # NAs
        TradeIbexLoader._validate_missing_ccontracts(contracts_df)

    @staticmethod
    def _validate_sources(trades_df: pd.DataFrame, contracts_df: pd.DataFrame):
        TradeIbexLoader._validate_trades_df(trades_df)
        TradeIbexLoader._validate_contracts_df(contracts_df)

    # BUILD
    @staticmethod
    def _build_database(
        trades_df: pd.DataFrame,
        contracts_df: pd.DataFrame,
        merge_columns: t.List[str],
        selected_columns_list: t.List[str],
    ) -> pd.DataFrame:

        # Merge
        merged_df = trades_df.merge(
            contracts_df, on=merge_columns, how="left", suffixes=("", "_contract")
        )

        # Add type of contract
        merged_df[config.data_config.merge_raw_config.contract_type_column] = merged_df[
            TradeIbexDatabaseEnum.CONTRACT_CODE.value
        ].apply(get_contract_type)

        # Select only relevant columns
        merged_df = merged_df[
            selected_columns_list
            + [config.data_config.merge_raw_config.contract_type_column]
        ]

        # Save CSV
        MERGE_RAW_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
        output_filename = config.data_config.merge_raw_config.output_filename
        output_file = MERGE_RAW_DATA_STEP_DIR_PATH / f"{output_filename}.csv"
        merged_df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

        logger.info(f"DF (with shape {merged_df.shape}) saved in: {output_file}.")

        return merged_df

    @staticmethod
    def load(
        merge_columns: t.List[str],
        selected_columns_list: t.List[str],
    ):
        trades_df, contracts_df = TradeIbexLoader._read_trades_and_contracts_dfs()
        TradeIbexLoader._validate_sources(trades_df, contracts_df)
        trade_ibex_db = TradeIbexLoader._build_database(
            trades_df, contracts_df, merge_columns, selected_columns_list
        )
        return trade_ibex_db
