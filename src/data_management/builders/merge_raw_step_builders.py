import logging

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH, config
from src.data_management.utils.contract_code_utils import (
    calculate_maturity_from_contract_code,
    get_contract_type,
)
from src.enums.data_enums import TradeIbexDatabaseEnum

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
        merge_columns = config.data_config.merge_raw_config.merge_columns_list
        selected_columns_list = (
            config.data_config.merge_raw_config.trade_ibex_columns_list
        )

        # Merge
        merged_df = tgentrades_df.merge(
            ccontracts_c2_df, on=merge_columns, how="left", suffixes=("", "_contract")
        )

        # Add type of contract
        merged_df[config.data_config.merge_raw_config.contract_type_column] = merged_df[
            TradeIbexDatabaseEnum.CONTRACT_CODE.value
        ].apply(get_contract_type)

        # Select only relevant columns
        merged_df = merged_df[
            selected_columns_list
            + [config.data_config.merge_raw_config.contract_type_column]
        ]

        # 1) Build series with calculated maturity from contract code + session date
        maturity_from_code = merged_df.apply(
            lambda row: calculate_maturity_from_contract_code(
                contract_code=row[TradeIbexDatabaseEnum.CONTRACT_CODE.value],
                session_date=row[TradeIbexDatabaseEnum.SESSION_DATE.value],
            ),
            axis=1,
        )

        # 2) Compare only where MaturityDate is NOT NA
        mask_has_maturity = merged_df[TradeIbexDatabaseEnum.MATURITY_DATE.value].notna()

        # Normaliza a date para comparar limpio (por si vienen como datetime/pandas Timestamp)
        existing = pd.to_datetime(
            merged_df.loc[mask_has_maturity, TradeIbexDatabaseEnum.MATURITY_DATE.value]
        ).dt.date
        computed = pd.to_datetime(maturity_from_code.loc[mask_has_maturity]).dt.date

        mismatch_mask = existing.ne(computed)

        # 3) If errors -> raise
        if mismatch_mask.any():
            bad_idx = existing.index[mismatch_mask]
            cols = [
                TradeIbexDatabaseEnum.SESSION_DATE.value,
                TradeIbexDatabaseEnum.CONTRACT_CODE.value,
                TradeIbexDatabaseEnum.MATURITY_DATE.value,
            ]
            sample = (
                merged_df.loc[bad_idx, cols]
                .assign(MaturityDateComputed=maturity_from_code.loc[bad_idx].values)
                .head(20)
            )
            raise ValueError(
                "MaturityDate mismatch detected (existing vs computed from ContractCode). "
                f"Total mismatches: {mismatch_mask.sum()}. Sample (up to 20):\n{sample}"
            )

        # 4) If ok -> impute NA in MaturityDate using computed series
        merged_df.loc[~mask_has_maturity, TradeIbexDatabaseEnum.MATURITY_DATE.value] = (
            maturity_from_code.loc[~mask_has_maturity].values
        )

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
