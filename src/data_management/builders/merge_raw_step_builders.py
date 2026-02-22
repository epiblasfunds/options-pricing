import logging

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH, config
from src.data_management.utils.contract_code_utils import get_contract_type
from src.enums.data_enums import CcontractsC2Enum, TgentradesEnum, TradeIbexDatabaseEnum

logger = logging.getLogger(__name__)


class TradeIbexBuilder:
    OUTPUT_FILENAME = (
        MERGE_RAW_DATA_STEP_DIR_PATH
        / f"{config.data_config.merge_raw_config.output_filename}.csv"
    )

    @staticmethod
    def get_output_filename():
        return TradeIbexBuilder.OUTPUT_FILENAME

    @staticmethod
    def build(tgentrades_df: pd.DataFrame, ccontracts_c2_df: pd.DataFrame):
        final_cols = [
            CcontractsC2Enum.CONTRACT_CODE.value,
            CcontractsC2Enum.STRIKE_PRICE.value,
            CcontractsC2Enum.MATURITY_DATE.value,
        ]
        ccontracts_c2_df = ccontracts_c2_df[final_cols]
        ccontracts_c2_df = ccontracts_c2_df.drop_duplicates(keep="first")

        

        # Impute MaturityDate and Strike for every contract that does not have it informed
        #   1. Extract unique ContractCodes in tgentrades_df
        #   2. Extract unique ContractCodes in ccontractcs_c2_df
        #   3. For every ContractCode in tgentrade_df which is not in ccontractcs_c2_df
        #       3.1. Calculate its MaturityDate
        #       3.2. Calculate its Strike
        #       3.3. Extend the ccontractcs_c2_df
        # TODO
        cc_tgentrades = tgentrades_df[TgentradesEnum.CONTRACT_CODE.value].drop_duplicates()
        

        selected_columns_list = (
            config.data_config.merge_raw_config.trade_ibex_columns_list
        )

        # Merge
        merge_columns = [
            TradeIbexDatabaseEnum.CONTRACT_CODE.value
        ]  # config.data_config.merge_raw_config.merge_columns_list
        merged_df = tgentrades_df.merge(ccontracts_c2_df, on=merge_columns, how="left")

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
        merged_df.to_csv(
            TradeIbexBuilder.get_output_filename(),
            index=False,
            encoding="utf-8",
            sep=";",
        )
        logger.info(
            f"DF (with shape {merged_df.shape}) saved in: {TradeIbexBuilder.get_output_filename()}."
        )

        return merged_df
