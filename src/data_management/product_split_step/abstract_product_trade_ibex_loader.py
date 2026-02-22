import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config
from src.data_management.merge_raw_step.trade_ibex_loader import TradeIbexLoader
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.enums.data_enums import ContractTypeEnum, TradeIbexDatabaseEnum
from src.exceptions.data_exceptions import (
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
)

logger = logging.getLogger(__name__)


class AbstractProductTradeIbexLoader(ABC):
    @classmethod
    @abstractmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        raise NotImplementedError("_get_contract_type not implemented.")

    @classmethod
    def get_output_filename(cls) -> Path:
        suffix = config.data_config.product_split_config.output_filename_contracts
        output_filename = f"{cls._get_contract_type().value}_{suffix}"
        return PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"

    # READ
    @staticmethod
    def _read_trade_ibex_database():
        trade_ibex_df = pd.read_csv(
            TradeIbexLoader.OUTPUT_FILENAME,
            delimiter=";",
            header=0,
            dtype="string",
        )
        return trade_ibex_df

    # VALIDATIONS
    @staticmethod
    def _validate_maturity(trade_ibex_df: pd.DataFrame, contract_type: ContractTypeEnum):
        contract_code_series = trade_ibex_df[TradeIbexDatabaseEnum.CONTRACT_CODE.value]
        maturity_series = trade_ibex_df[TradeIbexDatabaseEnum.MATURITY_DATE.value]
        session_date_series = trade_ibex_df[TradeIbexDatabaseEnum.SESSION_DATE.value]

        validate_maturity_contract_code(
            contract_type=contract_type,
            contract_code_series=contract_code_series,
            maturity_series=maturity_series,
            session_date_series=session_date_series,
        )

    @staticmethod
    def _validate_strike(trade_ibex_df: pd.DataFrame):
        contract_code_series = trade_ibex_df[TradeIbexDatabaseEnum.CONTRACT_CODE.value]
        strike_series = trade_ibex_df[TradeIbexDatabaseEnum.STRIKE_PRICE.value]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_sources(trade_ibex_df: pd.DataFrame):
        # Format validations
        if (trade_ibex_df["TradePrice"].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (trade_ibex_df["Quantity"].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()

        # Validate maturity with contract code
        TradeIbexLoader._validate_maturity(trade_ibex_df, ContractTypeEnum.OPTIONS)
        TradeIbexLoader._validate_maturity(trade_ibex_df, ContractTypeEnum.FUTURES)

        # Validate strikes with contract code
        TradeIbexLoader._validate_strike(trade_ibex_df)

        # NAs
        if trade_ibex_df.isna().any().any():
            raise MissingValuesError()

    # BUILD
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

    @staticmethod
    def load():
        trade_ibex_df = AbstractProductTradeIbexLoader._read_trades_and_contracts_dfs()
        AbstractProductTradeIbexLoader._validate_sources(trade_ibex_df)
        trade_ibex_db = AbstractProductTradeIbexLoader._build_database(trade_ibex_df)
        return trade_ibex_db
