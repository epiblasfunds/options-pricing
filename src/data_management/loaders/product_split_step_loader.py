import logging
import typing as t
from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config
from src.data_management.builders import (
    FutureTradesBuilder,
    OptionTradesBuilder,
    OptionUnderlyingBuilder,
)
from src.data_management.loaders.merge_raw_step_loader import MergeRawStepLoader
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.data_management.utils.data_type_utils import convert_data_types
from src.enums.data_enums import ContractTypeEnum, TradeIbexDBEnum
from src.exceptions.data_exceptions import (
    MissingValuesError,
    NegativeQuantityError,
    NegativeTimeToExpirationError,
    NegativeTradePriceError,
    SessionAfterMaturityError,
)

logger = logging.getLogger(__name__)


class ProductSplitStepLoader:
    @staticmethod
    def read_step_databases() -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        options_trade_ibex_df = pd.read_csv(
            OptionTradesBuilder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )
        futures_trade_ibex_df = pd.read_csv(
            FutureTradesBuilder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )
        options_underlying_ibex_df = pd.read_csv(
            OptionUnderlyingBuilder.get_output_filename(),
            delimiter=";",
            header=0,
            dtype="string",
        )

        options_trade_ibex_df = convert_data_types(
            df=options_trade_ibex_df,
            selected_columns_dict=config.data_config.product_split_config.options_trades_db_columns,
            format_date="%Y-%m-%d",
            format_datetime="%Y-%m-%d %H:%M:%S.%f",
        )
        futures_trade_ibex_df = convert_data_types(
            df=futures_trade_ibex_df,
            selected_columns_dict=config.data_config.product_split_config.futures_trades_db_columns,
            format_date="%Y-%m-%d",
            format_datetime="%Y-%m-%d %H:%M:%S.%f",
        )
        options_underlying_ibex_df = convert_data_types(
            df=options_underlying_ibex_df,
            selected_columns_dict=config.data_config.product_split_config.option_underlying_db_columns,
            format_datetime="%Y-%m-%d %H:%M:%S.%f",
        )

        return options_trade_ibex_df, futures_trade_ibex_df, options_underlying_ibex_df

    @classmethod
    def get_output_filename(cls) -> Path:
        suffix = config.data_config.product_split_config.output_filename_contracts
        output_filename = f"{cls._get_contract_type().name}_{suffix}"
        return PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"

    @staticmethod
    def _clean_futures_negative_expiration(trade_ibex_df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = trade_ibex_df.copy()

        futures_mask = (
            cleaned_df[TradeIbexDBEnum.CONTRACT_TYPE] == ContractTypeEnum.FUTURES
        )

        # Drop only futures with negative time to expiration
        time_to_expiration = pd.to_numeric(
            cleaned_df[TradeIbexDBEnum.TIME_TO_EXPIRATION], downcast="float"
        )
        negative_expiration_mask = futures_mask & (time_to_expiration < 0.0)

        if int(negative_expiration_mask.sum()) > 0:
            logger.info(
                "Dropping %s futures trades with negative TimeToExpiration.",
                int(negative_expiration_mask.sum()),
            )
            cleaned_df = cleaned_df.loc[~negative_expiration_mask].copy()

        return cleaned_df

    # VALIDATIONS
    @staticmethod
    def _validate_maturity(
        trade_ibex_df: pd.DataFrame, contract_type: ContractTypeEnum
    ):
        contract_code_series = trade_ibex_df[TradeIbexDBEnum.CONTRACT_CODE]
        maturity_series = trade_ibex_df[TradeIbexDBEnum.MATURITY_DATETIME]
        session_date_series = trade_ibex_df[TradeIbexDBEnum.SESSION_DATE]

        validate_maturity_contract_code(
            contract_type=contract_type,
            contract_code_series=contract_code_series,
            maturity_series=maturity_series,
            session_date_series=session_date_series,
        )

    @staticmethod
    def _validate_strike(trade_ibex_df: pd.DataFrame):
        contract_code_series = trade_ibex_df[TradeIbexDBEnum.CONTRACT_CODE]
        strike_series = trade_ibex_df[TradeIbexDBEnum.STRIKE_PRICE]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_source(trade_ibex_df: pd.DataFrame):
        # Format validations
        if (trade_ibex_df[TradeIbexDBEnum.TRADE_PRICE].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (trade_ibex_df[TradeIbexDBEnum.QUANTITY].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()

        if (
            trade_ibex_df[TradeIbexDBEnum.TIME_TO_EXPIRATION].astype("float64") < 0.0
        ).any():
            raise NegativeTimeToExpirationError()

        # Validate maturity with contract code
        ProductSplitStepLoader._validate_maturity(
            trade_ibex_df, ContractTypeEnum.OPTIONS
        )
        ProductSplitStepLoader._validate_maturity(
            trade_ibex_df, ContractTypeEnum.FUTURES
        )

        # Validate maturity and session date coherence
        session = pd.to_datetime(trade_ibex_df[TradeIbexDBEnum.SESSION_DATE])
        maturity = pd.to_datetime(trade_ibex_df[TradeIbexDBEnum.MATURITY_DATETIME])

        mask = session > maturity
        if mask.any():
            sample = trade_ibex_df[mask].iloc[0]
            raise SessionAfterMaturityError(
                f"SessionDate occurs after MaturityDate.\nExample: {sample}."
            )

        # Validate strikes with contract code
        ProductSplitStepLoader._validate_strike(trade_ibex_df)

        futures_mask = (
            trade_ibex_df[TradeIbexDBEnum.CONTRACT_TYPE] == ContractTypeEnum.FUTURES
        )
        futures_df = trade_ibex_df.loc[
            futures_mask,
            [c for c in trade_ibex_df.columns if c != TradeIbexDBEnum.STRIKE_PRICE],
        ]
        options_mask = (
            trade_ibex_df[TradeIbexDBEnum.CONTRACT_TYPE] == ContractTypeEnum.OPTIONS
        )
        options_df = trade_ibex_df[options_mask]
        if futures_df.isna().any().any() or options_df.isna().any().any():
            raise MissingValuesError()

    @staticmethod
    def load(force_reload=False) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if (
            force_reload
            or not OptionTradesBuilder.get_output_filename().exists()
            or not FutureTradesBuilder.get_output_filename().exists()
            or not OptionUnderlyingBuilder.get_output_filename().exists()
        ):
            trade_ibex_df = MergeRawStepLoader.load()
            trade_ibex_df = ProductSplitStepLoader._clean_futures_negative_expiration(
                trade_ibex_df
            )
            ProductSplitStepLoader._validate_source(trade_ibex_df)

            options_trade_ibex_db = OptionTradesBuilder.build(trade_ibex_df)
            futures_trade_ibex_db = FutureTradesBuilder.build(trade_ibex_df)
            OptionUnderlyingBuilder.build(
                options_trade_ibex_db=options_trade_ibex_db,
                futures_trade_ibex_db=futures_trade_ibex_db,
            )

        return ProductSplitStepLoader.read_step_databases()
