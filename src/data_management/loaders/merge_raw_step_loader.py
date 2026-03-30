import logging
import typing as t

import pandas as pd

from src.config.config import config
from src.data_management.builders import TradeIbexBuilder
from src.data_management.loaders.read_raw_step_loader import ReadRawStepLoader
from src.data_management.utils.contract_code_utils import (
    validate_maturity_contract_code,
    validate_strike_contract_code,
)
from src.data_management.utils.data_type_utils import convert_data_types
from src.enums.data_enums import CcontractsC2Enum, ContractTypeEnum, TgentradesEnum
from src.exceptions.data_exceptions import (
    DuplicatedPrimaryKeysError,
    MissingValuesError,
    NegativeQuantityError,
    NegativeTradePriceError,
)

logger = logging.getLogger(__name__)


class MergeRawStepLoader:
    @staticmethod
    def read_step_databases() -> pd.DataFrame:
        output_file = TradeIbexBuilder.get_output_filename()

        trade_ibex_df = pd.read_csv(
            output_file,
            delimiter=";",
            header=0,
            dtype="string",
        )
        trade_ibex_df = convert_data_types(
            df=trade_ibex_df,
            selected_columns_dict=config.data_config.merge_raw_config.trade_ibex_db_columns,
            format_date="%Y-%m-%d",
            format_datetime="%Y-%m-%d %H:%M:%S.%f",
        )
        return trade_ibex_df

    # VALIDATIONS
    @staticmethod
    def _validate_primary_keys(df: pd.DataFrame, pk_columns: t.List[str]):
        pk_df = df[pk_columns]
        dup_mask = pk_df.duplicated()
        if dup_mask.any():
            first_dup = pk_df[dup_mask].iloc[0]
            raise DuplicatedPrimaryKeysError(
                "MergeRawHandler::_validate_contracts_df. Duplicate (SessionDate, ContractCode) pair found: "
                f"SessionDate={first_dup[CcontractsC2Enum.SESSION_DATE]}, "
                f"ContractCode={first_dup[CcontractsC2Enum.CONTRACT_CODE]}."
            )

    @staticmethod
    def _validate_maturity(contracts_df: pd.DataFrame, contract_type: ContractTypeEnum):
        contract_code_series = contracts_df[CcontractsC2Enum.CONTRACT_CODE]
        maturity_series = contracts_df[CcontractsC2Enum.MATURITY_DATE]
        session_date_series = contracts_df[CcontractsC2Enum.SESSION_DATE]

        validate_maturity_contract_code(
            contract_type=contract_type,
            contract_code_series=contract_code_series,
            maturity_series=maturity_series,
            session_date_series=session_date_series,
        )

    @staticmethod
    def _validate_strike(contracts_df: pd.DataFrame):
        contract_code_series = contracts_df[CcontractsC2Enum.CONTRACT_CODE]
        strike_series = contracts_df[CcontractsC2Enum.STRIKE_PRICE]
        validate_strike_contract_code(
            contract_code_series=contract_code_series,
            strike_series=strike_series,
        )

    @staticmethod
    def _validate_missing_ccontracts(contracts_df: pd.DataFrame):
        cc_series = contracts_df[CcontractsC2Enum.CONTRACT_CODE]

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
            c for c in contracts_df.columns if c != CcontractsC2Enum.STRIKE_PRICE
        ]
        futures = contracts_df[futures_contracts_mask][future_columns]

        if options.isna().any().any() or futures.isna().any().any():
            raise MissingValuesError()

    @staticmethod
    def _validate_trades_df(trades_df):
        # Format validations
        if (trades_df[TgentradesEnum.TRADE_PRICE].astype("float64") <= 0.0).any():
            raise NegativeTradePriceError()

        if (trades_df[TgentradesEnum.QUANTITY].astype("float64") <= 0.0).any():
            raise NegativeQuantityError()

        # Unique Primary Keys
        MergeRawStepLoader._validate_primary_keys(
            df=trades_df,
            pk_columns=[
                TgentradesEnum.TRADE_EXEC_ID,
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
                CcontractsC2Enum.SESSION_DATE,
                CcontractsC2Enum.CONTRACT_CODE,
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
    def load(force_reload=False):
        if force_reload or not TradeIbexBuilder.get_output_filename().exists():
            contracts_df, trades_df, _ = ReadRawStepLoader.load()
            MergeRawStepLoader._validate_sources(trades_df, contracts_df)
            TradeIbexBuilder.build(trades_df, contracts_df)
        return MergeRawStepLoader.read_step_databases()
