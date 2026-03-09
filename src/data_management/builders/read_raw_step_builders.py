import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.config.config import RAW_DATA_STEP_DIR_PATH, config
from src.data_management.utils.contract_code_utils import (
    compute_ibex_mask,
    compute_monthly_maturity_mask,
)
from src.enums.data_enums import CcontractsC2Enum, TgentradesEnum
from src.enums.data_enums.rates_enum import RatesEnum

logger = logging.getLogger(__name__)


class AbstractMarketRawBuilder(ABC):
    @staticmethod
    @abstractmethod
    def get_output_filename() -> Path:
        raise NotImplementedError("get_output_filename not implemented.")

    @classmethod
    @abstractmethod
    def _get_contract_code_column(cls) -> str:
        raise NotImplementedError("_get_contract_type not implemented.")

    @classmethod
    @abstractmethod
    def _get_name(cls) -> str:
        raise NotImplementedError("_get_name not implemented.")

    @classmethod
    def build(cls, df: pd.DataFrame):
        # Filter by IBEX and monthly maturity
        contract_code_series = df[cls._get_contract_code_column()]

        ibex_mask = compute_ibex_mask(contract_code_series)
        monthly_maturity_mask = compute_monthly_maturity_mask(contract_code_series)
        filter_mask = ibex_mask & monthly_maturity_mask

        df = df[filter_mask]

        # Save CSV
        output_file = cls.get_output_filename()
        df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

        logger.info(
            f"{cls._get_name()} (with shape {df.shape}) saved in: {output_file}."
        )


class CContractsC2Builder(AbstractMarketRawBuilder):
    @staticmethod
    def get_output_filename() -> Path:
        return (
            RAW_DATA_STEP_DIR_PATH
            / f"{config.data_config.read_raw_config.cconctracts_c2_prefix}.csv"
        )

    @classmethod
    def _get_contract_code_column(cls) -> str:
        return CcontractsC2Enum.CONTRACT_CODE

    @classmethod
    def _get_name(cls) -> str:
        return "CContractsC2Database"


class TgentradesBuilder(AbstractMarketRawBuilder):
    @staticmethod
    def get_output_filename() -> Path:
        return (
            RAW_DATA_STEP_DIR_PATH
            / f"{config.data_config.read_raw_config.tgentrades_prefix}.csv"
        )

    @classmethod
    def _get_contract_code_column(cls) -> str:
        return TgentradesEnum.CONTRACT_CODE

    @classmethod
    def _get_name(cls) -> str:
        return "TgentradesDatabase"


class RatesBuilder:
    @staticmethod
    def get_output_filename() -> Path:
        return (
            RAW_DATA_STEP_DIR_PATH
            / f"{config.data_config.read_raw_config.rates_output_filename}.csv"
        )

    def build(
        eonia_df: pd.DataFrame,
        str_df: pd.DataFrame,
    ) -> pd.Series:

        # Set index
        rates_date_col_name = config.data_config.read_raw_config.rates_date_column_name
        eonia_df.set_index(rates_date_col_name, inplace=True)
        str_df.set_index(rates_date_col_name, inplace=True)
        
        idx_rate_values = config.data_config.read_raw_config.idx_rate_values
        eonia_series = eonia_df.iloc[:, idx_rate_values]
        str_series = str_df.iloc[:, idx_rate_values]

        # Rename columns
        eonia_series.columns = [RatesEnum.RATE]
        str_series.columns = [RatesEnum.RATE]
        eonia_series.rename_axis(RatesEnum.SESSION_DATE, inplace=True)
        str_series.rename_axis(RatesEnum.SESSION_DATE, inplace=True)

        # Take spread between STR and EONIA and cutoff date
        spread_str_eonia = config.data_config.read_raw_config.spread_str_eonia
        cutoff_date_str_eonia = config.data_config.read_raw_config.cutoff_date_str_eonia

        filtered_eonia_series = eonia_series[
            pd.to_datetime(eonia_series.index)
            < pd.to_datetime(cutoff_date_str_eonia)
        ].copy()
        filtered_eonia_series = pd.to_numeric(filtered_eonia_series) - spread_str_eonia
        filtered_str_series = str_series[
            pd.to_datetime(str_series.index)
            >= pd.to_datetime(cutoff_date_str_eonia)
        ].copy()
        filtered_str_series = pd.to_numeric(filtered_str_series)

        rates_series = pd.concat([filtered_eonia_series, filtered_str_series], ignore_index=False)
        rates_series.name = RatesEnum.RATE

        # Save CSV
        rates_series.to_csv(
            RatesBuilder.get_output_filename(),
            index=True,
            encoding="utf-8",
            sep=";",
        )
        logger.info(
            f"RatesDatabase (with shape {rates_series.shape}) saved in: {RatesBuilder.get_output_filename()}."
        )

        return rates_series
