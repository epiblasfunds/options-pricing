import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config
from src.enums.data_enums import (
    ContractTypeEnum,
    FuturesTradeIbexDBEnum,
    OptionsTradeIbexDBEnum,
)

logger = logging.getLogger(__name__)


class AbstractProductTradeIbexBuilder(ABC):
    @classmethod
    @abstractmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        raise NotImplementedError("_get_contract_type not implemented.")

    @classmethod
    @abstractmethod
    def _get_name(cls) -> str:
        raise NotImplementedError("_get_name not implemented.")

    @classmethod
    def get_output_filename(cls) -> Path:
        suffix = config.data_config.product_split_config.output_filename_contracts
        output_filename = f"{cls._get_contract_type()}_{suffix}"
        return PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"

    @classmethod
    def _build_database(cls, trade_ibex_df: pd.DataFrame) -> pd.DataFrame:
        contract_types_config = config.data_config.product_split_config.contract_types
        specific_cfg = contract_types_config[cls._get_contract_type()]

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

        logger.info(
            f"{cls._get_name()} (with shape {filtered_df.shape}) saved in: {output_file}."
        )

        return filtered_df


class OptionTradesBuilder(AbstractProductTradeIbexBuilder):
    @classmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        return ContractTypeEnum.OPTIONS

    @classmethod
    def _get_name(cls) -> str:
        return "OptionsTradeIbexDatabase"


class FutureTradesBuilder(AbstractProductTradeIbexBuilder):
    @classmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        return ContractTypeEnum.FUTURES

    @classmethod
    def _get_name(cls) -> str:
        return "FuturesTradeIbexDatabase"


class OptionUnderlyingBuilder:
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
                    OptionsTradeIbexDBEnum.OPTION_CONTRACT_CODE,
                    OptionsTradeIbexDBEnum.MATURITY_DATE,
                ]
            ]
            .drop_duplicates(
                subset=[OptionsTradeIbexDBEnum.OPTION_CONTRACT_CODE],
                keep="first",
            )
            .copy()
        )

        futures_df = (
            futures_trade_ibex_db[
                [
                    FuturesTradeIbexDBEnum.FUTURE_CONTRACT_CODE,
                    FuturesTradeIbexDBEnum.MATURITY_DATE,
                ]
            ]
            .drop_duplicates(
                subset=[FuturesTradeIbexDBEnum.FUTURE_CONTRACT_CODE],
                keep="first",
            )
            .copy()
        )

        # Merge on MaturityDate
        options_underlying_ibex_db = options_df.merge(
            futures_df,
            how="inner",  # Take only options with a matching future maturity
            on=OptionsTradeIbexDBEnum.MATURITY_DATE,
        )

        # Save CSV
        options_underlying_ibex_db.to_csv(
            OptionUnderlyingBuilder.get_output_filename(),
            index=False,
            encoding="utf-8",
            sep=";",
        )

        logger.info(
            f"OptionsUnderlyingIbexDatabase (with shape {options_underlying_ibex_db.shape}) saved in: "
            f"{OptionUnderlyingBuilder.get_output_filename()}."
        )

        return options_underlying_ibex_db
