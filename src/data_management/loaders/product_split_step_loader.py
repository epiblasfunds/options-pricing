import logging
import typing as t
from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config
from src.data_management.builders import (
    FuturesTradeIbexBuilder,
    OptionsTradeIbexBuilder,
    OptionsUnderlyingIbexBuilder,
    TradeIbexBuilder,
)
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


class ProductSplitStepLoader:
    @classmethod
    def get_output_filename(cls) -> Path:
        suffix = config.data_config.product_split_config.output_filename_contracts
        output_filename = f"{cls._get_contract_type().value}_{suffix}"
        return PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"

    # READ
    @staticmethod
    def _read_trade_ibex_database():
        trade_ibex_df = pd.read_csv(
            TradeIbexBuilder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )
        return trade_ibex_df

    # VALIDATIONS
    @staticmethod
    def _validate_maturity(
        trade_ibex_df: pd.DataFrame, contract_type: ContractTypeEnum
    ):
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
        ProductSplitStepLoader._validate_maturity(
            trade_ibex_df, ContractTypeEnum.OPTIONS
        )
        ProductSplitStepLoader._validate_maturity(
            trade_ibex_df, ContractTypeEnum.FUTURES
        )

        # Validate strikes with contract code
        ProductSplitStepLoader._validate_strike(trade_ibex_df)

        # NAs
        if trade_ibex_df.isna().any().any():
            raise MissingValuesError()

    @staticmethod
    def load() -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        trade_ibex_df = ProductSplitStepLoader._read_trade_ibex_database()
        ProductSplitStepLoader._validate_sources(trade_ibex_df)

        options_trade_ibex_db = OptionsTradeIbexBuilder._build_database(trade_ibex_df)
        futures_trade_ibex_db = FuturesTradeIbexBuilder._build_database(trade_ibex_df)
        options_underlying_ibex_db = OptionsUnderlyingIbexBuilder.build(
            options_trade_ibex_db=options_trade_ibex_db,
            futures_trade_ibex_db=futures_trade_ibex_db,
        )

        return options_trade_ibex_db, futures_trade_ibex_db, options_underlying_ibex_db
