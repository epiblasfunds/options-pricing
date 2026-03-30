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
    @abstractmethod
    def _to_csv(cls, df: pd.DataFrame) -> str:
        raise NotImplementedError("_to_csv not implemented.")

    @classmethod
    def get_output_filename(cls) -> Path:
        suffix = config.data_config.product_split_config.output_filename_contracts
        output_filename = f"{cls._get_contract_type().name}_{suffix}"
        return PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"

    @classmethod
    def _build_database(cls, trade_ibex_df: pd.DataFrame) -> pd.DataFrame:
        contract_types_config = config.data_config.product_split_config.contract_types
        specific_cfg = contract_types_config[cls._get_contract_type()]

        schema_by_contract_type = {
            ContractTypeEnum.OPTIONS: config.data_config.product_split_config.options_trades_db_columns,
            ContractTypeEnum.FUTURES: config.data_config.product_split_config.futures_trades_db_columns,
        }
        output_columns = list(schema_by_contract_type[cls._get_contract_type()].keys())

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
        filtered_df = filtered_df[output_columns]

        # Save CSV
        cls._to_csv(filtered_df)

        return filtered_df


class OptionTradesBuilder(AbstractProductTradeIbexBuilder):
    @classmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        return ContractTypeEnum.OPTIONS

    @classmethod
    def _get_name(cls) -> str:
        return "OptionsTradeIbexDatabase"

    @classmethod
    def build(cls, trade_ibex_df: pd.DataFrame) -> pd.DataFrame:
        return cls._build_database(trade_ibex_df=trade_ibex_df)

    @classmethod
    def _to_csv(cls, df: pd.DataFrame):
        df_copy = df.copy()

        # Format Datetimes
        for col in [
            OptionsTradeIbexDBEnum.EXEC_DATETIME,
            OptionsTradeIbexDBEnum.MATURITY_DATETIME,
        ]:
            df_copy[col] = df_copy[col].dt.strftime(date_format="%Y-%m-%d %H:%M:%S.%f")

        df_copy.to_csv(
            cls.get_output_filename(),
            encoding="utf-8",
            sep=";",
            index=False,
        )

        logger.info(
            f"OptionsTradeIbexDatabase (with shape {df_copy.shape}) "
            + f"saved in: {cls.get_output_filename()}."
        )


class FutureTradesBuilder(AbstractProductTradeIbexBuilder):
    @classmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        return ContractTypeEnum.FUTURES

    @classmethod
    def _get_name(cls) -> str:
        return "FuturesTradeIbexDatabase"

    @classmethod
    def build(cls, trade_ibex_df: pd.DataFrame) -> pd.DataFrame:
        return cls._build_database(trade_ibex_df=trade_ibex_df)

    @classmethod
    def _to_csv(cls, df: pd.DataFrame):
        df_copy = df.copy()

        # Format Datetimes
        for col in [
            FuturesTradeIbexDBEnum.EXEC_DATETIME,
            FuturesTradeIbexDBEnum.MATURITY_DATETIME,
        ]:
            df_copy[col] = df_copy[col].dt.strftime(date_format="%Y-%m-%d %H:%M:%S.%f")

        df_copy.to_csv(
            cls.get_output_filename(),
            encoding="utf-8",
            sep=";",
            index=False,
        )

        logger.info(
            f"FuturesTradeIbexDatabase (with shape {df_copy.shape}) "
            + f"saved in: {cls.get_output_filename()}."
        )


class OptionUnderlyingBuilder:
    @staticmethod
    def get_output_filename() -> Path:
        output_filename = (
            config.data_config.product_split_config.output_filename_relationship
        )
        return PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"

    @classmethod
    def _to_csv(cls, df: pd.DataFrame):
        # Format Datetimes
        for col in [
            OptionsTradeIbexDBEnum.MATURITY_DATETIME,
        ]:
            df[col] = df[col].dt.strftime(date_format="%Y-%m-%d %H:%M:%S.%f")

        df.to_csv(
            cls.get_output_filename(),
            encoding="utf-8",
            sep=";",
            index=False,
        )

        logger.info(
            f"OptionsUnderlyingDB (with shape {df.shape}) "
            + f"saved in: {cls.get_output_filename()}."
        )

    @classmethod
    def build(
        cls, options_trade_ibex_db: pd.DataFrame, futures_trade_ibex_db: pd.DataFrame
    ):
        # Select relevant columns
        options_df = (
            options_trade_ibex_db[
                [
                    OptionsTradeIbexDBEnum.OPTION_CONTRACT_CODE,
                    OptionsTradeIbexDBEnum.MATURITY_DATETIME,
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
                    FuturesTradeIbexDBEnum.MATURITY_DATETIME,
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
            on=OptionsTradeIbexDBEnum.MATURITY_DATETIME,
        )

        # Save CSV
        cls._to_csv(options_underlying_ibex_db)

        return options_underlying_ibex_db
