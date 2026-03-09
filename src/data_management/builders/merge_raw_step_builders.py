import logging

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH, config
from src.data_management.utils.contract_code_utils import (
    calculalte_strike_from_contract_code,
    calculate_maturity_from_contract_code,
    get_contract_type,
)
from src.enums.data_enums import (
    CcontractsC2Enum,
    ContractTypeEnum,
    TgentradesEnum,
    TradeIbexDBEnum,
)

logger = logging.getLogger(__name__)


class TradeIbexBuilder:
    OUTPUT_FILENAME = (
        MERGE_RAW_DATA_STEP_DIR_PATH
        / f"{config.data_config.merge_raw_config.output_filename}.csv"
    )
    SESSION_DATE_YEAR_COLUMN = f"{TradeIbexDBEnum.SESSION_DATE}Year"

    @staticmethod
    def get_output_filename():
        return TradeIbexBuilder.OUTPUT_FILENAME

    @staticmethod
    def impute_missing_strikes(merged_df: pd.DataFrame) -> pd.DataFrame:
        # 2026-02-23 18:09:24,906 | INFO | src.data_management.builders.merge_raw_step_builders |
        # There are 16442721 missing maturities corresponding to 4600 contract codes.
        mask_missing_strike_price = merged_df[TradeIbexDBEnum.STRIKE_PRICE].isna()
        mask_options = (
            merged_df[TradeIbexDBEnum.CONTRACT_TYPE] == ContractTypeEnum.OPTIONS
        )
        mask = mask_missing_strike_price & mask_options

        missing_cc = merged_df.loc[
            mask,
            [
                TradeIbexDBEnum.CONTRACT_CODE,
            ],
        ].drop_duplicates()

        logger.info(
            f"There are {mask.sum()} option trades with missing strike price,"
            f"corresponding to {missing_cc.shape[0]} different {TradeIbexDBEnum.CONTRACT_CODE}."
            " Imputing missing strikes."
        )

        # Compute strikes
        missing_cc[TradeIbexDBEnum.STRIKE_PRICE] = missing_cc.apply(
            lambda row: str(
                calculalte_strike_from_contract_code(
                    contract_code=row[TradeIbexDBEnum.CONTRACT_CODE],
                )
            ),
            axis=1,
        )

        # Imputing: Merge + fillna + drop extracolumn
        suffix = "_from_missings_cc"
        column_to_drop = TradeIbexDBEnum.STRIKE_PRICE + suffix

        # Merge
        merged_df = merged_df.merge(
            missing_cc,
            on=[
                TradeIbexDBEnum.CONTRACT_CODE,
            ],
            how="left",
            suffixes=("", suffix),
        )

        # Fillna
        merged_df[TradeIbexDBEnum.STRIKE_PRICE] = merged_df[
            TradeIbexDBEnum.STRIKE_PRICE
        ].fillna(merged_df[column_to_drop])

        # Drop
        merged_df = merged_df.drop(columns=[column_to_drop])

        return merged_df

    @staticmethod
    def impute_missing_maturities(merged_df: pd.DataFrame) -> pd.DataFrame:
        # 2026-02-23 18:09:24,906 | INFO | src.data_management.builders.merge_raw_step_builders |
        # There are 16442721 missing maturities corresponding to 4600 contract codes.
        mask = merged_df[TradeIbexDBEnum.MATURITY_DATETIME].isna()
        missing_cc = merged_df.loc[
            mask,
            [
                TradeIbexDBEnum.SESSION_DATE,
                TradeIbexDBEnum.CONTRACT_CODE,
            ],
        ].drop_duplicates()

        logger.info(
            f"There are {mask.sum()} trades with missing maturities,"
            f"corresponding to {missing_cc.shape[0]} different "
            f"({TradeIbexDBEnum.SESSION_DATE}, {TradeIbexDBEnum.CONTRACT_CODE}) pairs. "
            "Imputing missing maturities based on this pair."
        )

        # Compute maturities
        missing_cc[TradeIbexDBEnum.MATURITY_DATETIME] = missing_cc.apply(
            lambda row: calculate_maturity_from_contract_code(
                contract_code=row[TradeIbexDBEnum.CONTRACT_CODE],
                session_date=pd.to_datetime(row[TradeIbexDBEnum.SESSION_DATE]).date(),
            ).strftime("%Y-%m-%d"),
            axis=1,
        )

        # Imputing: Merge + fillna + drop extracolumn
        suffix = "_from_missings_cc"
        column_to_drop = TradeIbexDBEnum.MATURITY_DATETIME + suffix

        # Merge
        merged_df = merged_df.merge(
            missing_cc,
            on=[
                TradeIbexDBEnum.SESSION_DATE,
                TradeIbexDBEnum.CONTRACT_CODE,
            ],
            how="left",
            suffixes=("", suffix),
        )

        # Fillna
        merged_df[TradeIbexDBEnum.MATURITY_DATETIME] = merged_df[
            TradeIbexDBEnum.MATURITY_DATETIME
        ].fillna(merged_df[column_to_drop])

        # Drop
        merged_df = merged_df.drop(columns=[column_to_drop])

        return merged_df

    @staticmethod
    def create_exec_datetime(
        df: pd.DataFrame, exec_time_col: str, session_date_col: str
    ) -> pd.Series:
        # Extract only the time component in case the column contains a full datetime
        df[exec_time_col] = df[exec_time_col].astype(str).str.split().str[-1]

        # Ensure microseconds are present by appending ".000000" when missing
        df[exec_time_col] = df[exec_time_col].apply(
            lambda x: x if "." in x else x + ".000000"
        )

        # Combine date and time and convert to pandas datetime
        new_col = pd.to_datetime(
            df[session_date_col].astype(str) + " " + df[exec_time_col],
            format="%Y-%m-%d %H:%M:%S.%f",
        )

        df[TradeIbexDBEnum.EXEC_DATETIME] = new_col

        return df

    @staticmethod
    def create_time_to_expiration_column(df: pd.DataFrame) -> pd.DataFrame:

        # Convert time columns to datetime
        df[TradeIbexDBEnum.EXEC_DATETIME] = pd.to_datetime(
            df[TradeIbexDBEnum.EXEC_DATETIME], format="ISO8601"
        )
        df[TradeIbexDBEnum.MATURITY_DATETIME] = pd.to_datetime(
            df[TradeIbexDBEnum.MATURITY_DATETIME].astype(str)
            + config.data_config.merge_raw_config.maturity_hour_expiration,
            format="%Y-%m-%d %H:%M:%S.%f",
        )

        # Calculate time to expiration in days with decimals
        time_delta = (
            df[TradeIbexDBEnum.MATURITY_DATETIME] - df[TradeIbexDBEnum.EXEC_DATETIME]
        )
        df[TradeIbexDBEnum.TIME_TO_EXPIRATION] = time_delta.dt.total_seconds() / (
            24 * 3600
        )

        return df

    @staticmethod
    def create_new_columns(df: pd.DataFrame) -> pd.DataFrame:

        df = TradeIbexBuilder.create_exec_datetime(
            df=df,
            exec_time_col=TgentradesEnum.EXEC_TIME,
            session_date_col=TgentradesEnum.SESSION_DATE,
        )
        df = TradeIbexBuilder.create_time_to_expiration_column(df=df)

        return df

    @staticmethod
    def build(
        tgentrades_df: pd.DataFrame, ccontracts_c2_df: pd.DataFrame
    ) -> pd.DataFrame:
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
            ccontracts_c2_df[CcontractsC2Enum.SESSION_DATE]
        ).dt.year.astype("str")

        tgentrades_df[TradeIbexBuilder.SESSION_DATE_YEAR_COLUMN] = pd.to_datetime(
            tgentrades_df[TgentradesEnum.SESSION_DATE]
        ).dt.year.astype("str")

        final_cols = [
            CcontractsC2Enum.CONTRACT_CODE,
            CcontractsC2Enum.STRIKE_PRICE,
            CcontractsC2Enum.MATURITY_DATE,
            TradeIbexBuilder.SESSION_DATE_YEAR_COLUMN,
        ]
        ccontracts_c2_df = ccontracts_c2_df[final_cols]
        ccontracts_c2_df = ccontracts_c2_df.drop_duplicates(keep="first")

        # Merge
        merge_columns = [
            TradeIbexDBEnum.CONTRACT_CODE,
            TradeIbexBuilder.SESSION_DATE_YEAR_COLUMN,
        ]
        merged_df = tgentrades_df.merge(ccontracts_c2_df, on=merge_columns, how="left")
        merged_df = merged_df.rename(
            columns={CcontractsC2Enum.MATURITY_DATE: TradeIbexDBEnum.MATURITY_DATETIME}
        )

        # Add type of contract
        merged_df[config.data_config.merge_raw_config.contract_type_column] = merged_df[
            TradeIbexDBEnum.CONTRACT_CODE
        ].apply(get_contract_type)

        # Impute missing maturity dates and strikes
        merged_df = TradeIbexBuilder.impute_missing_maturities(merged_df=merged_df)
        merged_df = TradeIbexBuilder.impute_missing_strikes(merged_df=merged_df)

        # Select only relevant columns
        merged_df = merged_df[
            config.data_config.merge_raw_config.trade_ibex_columns_list
            + [config.data_config.merge_raw_config.contract_type_column]
        ]

        # Create new columns
        merged_df = TradeIbexBuilder.create_new_columns(df=merged_df)

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
