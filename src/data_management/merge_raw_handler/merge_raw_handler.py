import logging
import typing as t
from pathlib import Path

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH, config
from src.data_management.merge_raw_handler.get_contract_type_handler import (
    get_contract_type,
)
from src.enums.data_enums.trade_ibex_database_enum import TradeIbexDatabaseEnum
from src.exceptions.data_exceptions import DataError

logger = logging.getLogger(__name__)


class MergeRawHandler:
    def __init__(
        self,
        trades_filename: str,
        contracts_filename: str,
        merge_columns: t.List[str],
        selected_columns_list: t.List[str],
    ):
        self._read_trades_and_contracts_dfs(trades_filename, contracts_filename)
        self._validate_sources()
        self._build(merge_columns, selected_columns_list)

    # READ
    def _read_trades_and_contracts_dfs(
        self, trades_filename: str, contracts_filename: str
    ) -> None:
        self.trades_df = pd.read_csv(
            Path(trades_filename),
            delimiter=";",
            header=0,
            dtype="string",
        )
        self.contracts_df = pd.read_csv(
            Path(contracts_filename),
            delimiter=";",
            header=0,
            dtype="string",
        )

    # VALIDATIONS
    def _validate_trades_df(self):
        # Format validations
        if (self.trades_df["ContractCode"].str.len() != 16).any():
            raise DataError(
                "MergeRawHandler::_validate_trades_df. There are rows with non expected len in ContractCode column."
            )
        if (self.trades_df["MarketCode"].str.len() != 2).any():
            raise DataError(
                "MergeRawHandler::_validate_trades_df. There are rows with non expected len in MarketCode column."
            )
        if (self.trades_df["TradeExecID"].str.len() != 12).any():
            raise DataError(
                "MergeRawHandler::_validate_trades_df. There are rows with non expected len in TradeExecID column."
            )
        if (self.trades_df["TradeType"].str.len() != 1).any():
            raise DataError(
                "MergeRawHandler::_validate_trades_df. There are rows with non expected len in TradeType column."
            )

        # Positive number validations
        if (self.trades_df["TradePrice"].astype("float64") > 0.0).any():
            raise DataError(
                "MergeRawHandler::_validate_trades_df. There are rows with negative values in TradePrice column."
            )
        if (self.trades_df["Quantity"].astype("float64") > 0.0).any():
            raise DataError(
                "MergeRawHandler::_validate_trades_df. There are rows with negative values in TradePrice column."
            )

        # Unique Primary Keys
        dup_mask = self.trades_df.duplicated(subset=["SessionDate", "ContractCode"])
        if dup_mask.any():
            first_dup = self.trades_df.loc[
                dup_mask, ["SessionDate", "ContractCode"]
            ].iloc[0]
            raise DataError(
                "MergeRawHandler::_validate_trades_df. Duplicate (SessionDate, ContractCode) pair found: "
                f"SessionDate={first_dup['SessionDate']}, ContractCode={first_dup['ContractCode']}."
            )

        # NAs
        if self.trades_df.isna().any().any():
            raise DataError(
                "MergeRawHandler::_validate_trades_df. There are NAs in trades df."
            )

    def _validate_contracts_df(self):
        # Validate that the maturity extracted from the contract code
        # is the same that the MaturityDate column
        maturity_calculated = ""

        # Unique Primary Keys
        # TODO

        # Same contract code has same maturities and same strikes
        # TODO

        # NAs
        if self.trades_df.isna().any().any():
            raise DataError(
                "MergeRawHandler::_validate_trades_df. There are NAs in trades df."
            )

    def _validate_sources(self):
        self._validate_trades_df()
        self._validate_contracts_df()

    # BUILD
    def _build_database(
        self,
        merge_columns: t.List[str],
        selected_columns_list: t.List[str],
    ) -> pd.DataFrame:

        # Merge
        merged_df = self.trades_df.merge(
            self.contracts_df, on=merge_columns, how="left", suffixes=("", "_contract")
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
