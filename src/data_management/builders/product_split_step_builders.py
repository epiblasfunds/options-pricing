import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config
from src.enums.data_enums import ContractTypeEnum, OptionsTradeIbexDatabaseEnum
from src.enums.data_enums.futures_trade_ibex_database_enum import (
    FuturesTradeIbexDatabaseEnum,
)

logger = logging.getLogger(__name__)


class AbstractProductTradeIbexBuilder(ABC):
    @classmethod
    @abstractmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        raise NotImplementedError("_get_contract_type not implemented.")

    @classmethod
    def get_output_filename(cls) -> Path:
        suffix = config.data_config.product_split_config.output_filename_contracts
        output_filename = f"{cls._get_contract_type().value}_{suffix}"
        return PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"

    @classmethod
    def _build_database(cls, trade_ibex_df: pd.DataFrame) -> pd.DataFrame:
        contract_types_config = config.data_config.product_split_config.contract_types
        specific_cfg = contract_types_config[cls._get_contract_type().value]

        filter_col = config.data_config.product_split_config.filter_contract_column
        filtered_df = trade_ibex_df[
            trade_ibex_df[filter_col].str.startswith(
                tuple(specific_cfg["prefixes"]), na=False
            )
        ].copy()
        filtered_df.rename(
            columns={filter_col: specific_cfg["contract_column_new"]},
            inplace=True,
        )

        # Select columns
        filtered_df = filtered_df[specific_cfg["columns"]]

        # Save CSV
        output_file = cls.get_output_filename()
        filtered_df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

        logger.info(f"DF (with shape {filtered_df.shape}) saved in: {output_file}.")

        return filtered_df


class OptionsTradeIbexBuilder(AbstractProductTradeIbexBuilder):
    @classmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        return ContractTypeEnum.OPTIONS


class FuturesTradeIbexBuilder(AbstractProductTradeIbexBuilder):
    @classmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        return ContractTypeEnum.FUTURES


class OptionsUnderlyingIbexBuilder:
    @staticmethod
    def get_output_filename() -> Path:
        output_filename = (
            config.data_config.product_split_config.output_filename_relationship
        )
        return PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"

    @staticmethod
    def build(options_trade_ibex_db: pd.DataFrame, futures_trade_ibex_db: pd.DataFrame):
        # Select relevant columns
        options_df = (
            options_trade_ibex_db[
                [
                    OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value,
                    OptionsTradeIbexDatabaseEnum.MATURITY_DATE.value,
                ]
            ]
            .drop_duplicates(
                subset=[OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value],
                keep="first",
            )
            .copy()
        )

        futures_df = (
            futures_trade_ibex_db[
                [
                    FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE.value,
                    FuturesTradeIbexDatabaseEnum.MATURITY_DATE.value,
                ]
            ]
            .drop_duplicates(
                subset=[FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE.value],
                keep="first",
            )
            .copy()
        )

        # Merge on MaturityDate
        options_underlying_ibex_db = options_df.merge(
            futures_df,
            how="left",
            on=OptionsTradeIbexDatabaseEnum.MATURITY_DATE.value,
        )

        # Save CSV
        options_underlying_ibex_db.to_csv(
            OptionsUnderlyingIbexBuilder.get_output_filename(),
            index=False,
            encoding="utf-8",
            sep=";",
        )

        logger.info(
            f"DF (with shape {options_underlying_ibex_db.shape}) saved in: "
            f"{OptionsUnderlyingIbexBuilder.get_output_filename()}."
        )

        return options_underlying_ibex_db
