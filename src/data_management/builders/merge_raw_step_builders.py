import logging

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH, config
from src.data_management.utils.contract_code_utils import (
    calculalte_strike_from_contract_code,
    calculate_maturity_from_contract_code,
    get_contract_type,
)
from src.enums.data_enums import CcontractsC2Enum, TgentradesEnum, TradeIbexDatabaseEnum
from src.enums.data_enums.contract_type_enum import ContractTypeEnum

logger = logging.getLogger(__name__)


class TradeIbexBuilder:
    OUTPUT_FILENAME = (
        MERGE_RAW_DATA_STEP_DIR_PATH
        / f"{config.data_config.merge_raw_config.output_filename}.csv"
    )
    SESSION_DATE_YEAR_COLUMN = f"{TradeIbexDatabaseEnum.SESSION_DATE.value}Year"

    @staticmethod
    def get_output_filename():
        return TradeIbexBuilder.OUTPUT_FILENAME

    @staticmethod
    def impute_missing_strikes(merged_df: pd.DataFrame) -> pd.DataFrame:
        # 2026-02-23 18:09:24,906 | INFO | src.data_management.builders.merge_raw_step_builders |
        # There are 16442721 missing maturities corresponding to 4600 contract codes.
        mask_missing_strike_price = merged_df[
            TradeIbexDatabaseEnum.STRIKE_PRICE.value
        ].isna()
        mask_options = (
            merged_df[TradeIbexDatabaseEnum.CONTRACT_TYPE.value]
            == ContractTypeEnum.OPTIONS.value
        )
        mask = mask_missing_strike_price & mask_options

        missing_cc = merged_df.loc[
            mask,
            [
                TradeIbexDatabaseEnum.CONTRACT_CODE.value,
            ],
        ].drop_duplicates()

        logger.info(
            f"There are {mask.sum()} option trades with missing strike price,"
            f"corresponding to {missing_cc.shape[0]} different {TradeIbexDatabaseEnum.CONTRACT_CODE.value}."
            "Imputing missing strikes."
        )

        # Compute strikes
        missing_cc[TradeIbexDatabaseEnum.STRIKE_PRICE.value] = missing_cc.apply(
            lambda row: str(
                calculalte_strike_from_contract_code(
                    contract_code=row[TradeIbexDatabaseEnum.CONTRACT_CODE.value],
                )
            ),
            axis=1,
        )

        # Imputing: Merge + fillna + drop extracolumn
        suffix = "_from_missings_cc"
        column_to_drop = TradeIbexDatabaseEnum.STRIKE_PRICE.value + suffix

        # Merge
        merged_df = merged_df.merge(
            missing_cc,
            on=[
                TradeIbexDatabaseEnum.CONTRACT_CODE.value,
            ],
            how="left",
            suffixes=("", suffix),
        )

        # Fillna
        merged_df[TradeIbexDatabaseEnum.STRIKE_PRICE.value] = merged_df[
            TradeIbexDatabaseEnum.STRIKE_PRICE.value
        ].fillna(merged_df[column_to_drop])

        # Drop
        merged_df = merged_df.drop(columns=[column_to_drop])

        return merged_df

    @staticmethod
    def impute_missing_maturities(merged_df: pd.DataFrame) -> pd.DataFrame:
        # 2026-02-23 18:09:24,906 | INFO | src.data_management.builders.merge_raw_step_builders |
        # There are 16442721 missing maturities corresponding to 4600 contract codes.
        mask = merged_df[TradeIbexDatabaseEnum.MATURITY_DATE.value].isna()
        missing_cc = merged_df.loc[
            mask,
            [
                TradeIbexDatabaseEnum.SESSION_DATE.value,
                TradeIbexDatabaseEnum.CONTRACT_CODE.value,
            ],
        ].drop_duplicates()

        logger.info(
            f"There are {mask.sum()} trades with missing maturities,"
            f"corresponding to {missing_cc.shape[0]} different "
            f"({TradeIbexDatabaseEnum.SESSION_DATE.value}, {TradeIbexDatabaseEnum.CONTRACT_CODE.value}) pairs."
            "Imputing missing maturities based on this pair."
        )

        # Compute maturities
        missing_cc[TradeIbexDatabaseEnum.MATURITY_DATE.value] = missing_cc.apply(
            lambda row: calculate_maturity_from_contract_code(
                contract_code=row[TradeIbexDatabaseEnum.CONTRACT_CODE.value],
                session_date=pd.to_datetime(
                    row[TradeIbexDatabaseEnum.SESSION_DATE.value]
                ).date(),
            ).strftime("%Y-%m-%d"),
            axis=1,
        )

        # Imputing: Merge + fillna + drop extracolumn
        suffix = "_from_missings_cc"
        column_to_drop = TradeIbexDatabaseEnum.MATURITY_DATE.value + suffix

        # Merge
        merged_df = merged_df.merge(
            missing_cc,
            on=[
                TradeIbexDatabaseEnum.SESSION_DATE.value,
                TradeIbexDatabaseEnum.CONTRACT_CODE.value,
            ],
            how="left",
            suffixes=("", suffix),
        )

        # Fillna
        merged_df[TradeIbexDatabaseEnum.MATURITY_DATE.value] = merged_df[
            TradeIbexDatabaseEnum.MATURITY_DATE.value
        ].fillna(merged_df[column_to_drop])

        # Drop
        merged_df = merged_df.drop(columns=[column_to_drop])

        return merged_df

    @staticmethod
    def build(tgentrades_df: pd.DataFrame, ccontracts_c2_df: pd.DataFrame):
        """
        # This building process requires to merge to dataframes and to
        # fill missing values. That means that the process has to:
        #   - Optimize the memory allocation: merge can lead to allocate too much memory.
        #       For this reason, the CContractsC2 df will be reduced. With:
        #           + Year of the session date (for calculating missing maturities in futures)
        #           + Contract Code
        #           + Maturity Date
        #           + Strike Price
        #       we have all the info required for building the TradeIbexDB.
        #   - Optimize processing time: there is no need to calculate the maturity for every
        #       trade. It is enough to calculate the maturity
        #       for every (SessionDate, ContractCode) that are actually missing.
        """
        ccontracts_c2_df[TradeIbexBuilder.SESSION_DATE_YEAR_COLUMN] = pd.to_datetime(
            ccontracts_c2_df[CcontractsC2Enum.SESSION_DATE.value]
        ).dt.year.astype("str")

        tgentrades_df[TradeIbexBuilder.SESSION_DATE_YEAR_COLUMN] = pd.to_datetime(
            tgentrades_df[TgentradesEnum.SESSION_DATE.value]
        ).dt.year.astype("str")

        final_cols = [
            CcontractsC2Enum.CONTRACT_CODE.value,
            CcontractsC2Enum.STRIKE_PRICE.value,
            CcontractsC2Enum.MATURITY_DATE.value,
            TradeIbexBuilder.SESSION_DATE_YEAR_COLUMN,
        ]
        ccontracts_c2_df = ccontracts_c2_df[final_cols]
        ccontracts_c2_df = ccontracts_c2_df.drop_duplicates(keep="first")

        # Merge
        merge_columns = [
            TradeIbexDatabaseEnum.CONTRACT_CODE.value,
            TradeIbexBuilder.SESSION_DATE_YEAR_COLUMN,
        ]  # config.data_config.merge_raw_config.merge_columns_list
        merged_df = tgentrades_df.merge(ccontracts_c2_df, on=merge_columns, how="left")

        # Add type of contract
        merged_df[config.data_config.merge_raw_config.contract_type_column] = merged_df[
            TradeIbexDatabaseEnum.CONTRACT_CODE.value
        ].apply(get_contract_type)

        # Impute missing maturity dates and strikes
        merged_df = TradeIbexBuilder.impute_missing_maturities(merged_df=merged_df)
        merged_df = TradeIbexBuilder.impute_missing_strikes(merged_df=merged_df)

        # Select only relevant columns
        merged_df = merged_df[
            config.data_config.merge_raw_config.trade_ibex_columns_list
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
            f"TradeIbexDatabase (with shape {merged_df.shape}) saved in: {TradeIbexBuilder.get_output_filename()}."
        )

        return merged_df
