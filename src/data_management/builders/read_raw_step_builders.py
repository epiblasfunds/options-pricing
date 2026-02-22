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

logger = logging.getLogger(__name__)


class AbstractRawBuilder(ABC):
    @classmethod
    @abstractmethod
    def _get_contract_code_column(cls) -> str:
        raise NotImplementedError("_get_contract_type not implemented.")

    @staticmethod
    @abstractmethod
    def get_output_filename() -> Path:
        raise NotImplementedError("get_output_filename not implemented.")

    @classmethod
    def build(cls, tgentrades_df: pd.DataFrame):
        # Filter by IBEX and monthly maturity
        contract_code_series = tgentrades_df[cls._get_contract_code_column()]

        ibex_mask = compute_ibex_mask(contract_code_series)
        monhtly_maturity_mask = compute_monthly_maturity_mask(contract_code_series)
        filter_mask = ibex_mask & monhtly_maturity_mask

        tgentrades_df = tgentrades_df[filter_mask]

        # Save CSV
        output_file = cls.get_output_filename()
        tgentrades_df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

        logger.info(f"DF (with shape {tgentrades_df.shape}) saved in: {output_file}.")


class CContractsC2Builder(AbstractRawBuilder):
    @staticmethod
    def get_output_filename() -> Path:
        return (
            RAW_DATA_STEP_DIR_PATH
            / f"{config.data_config.read_raw_config.cconctracts_c2_prefix}.csv"
        )

    @classmethod
    def _get_contract_code_column(cls) -> str:
        return CcontractsC2Enum.CONTRACT_CODE.value


class TgentradesBuilder(AbstractRawBuilder):
    @staticmethod
    def get_output_filename() -> Path:
        return (
            RAW_DATA_STEP_DIR_PATH
            / f"{config.data_config.read_raw_config.tgentrades_prefix}.csv"
        )

    @classmethod
    def _get_contract_code_column(cls) -> str:
        return TgentradesEnum.CONTRACT_CODE.value
