import logging
import typing as t
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config.config import RAW_DATA_STEP_DIR_PATH, SOURCE_DATA_DIR_PATH, config
from src.enums.data_enums.ccontracts_c2_enum import CcontractsC2Enum
from src.enums.data_enums.data_type_enum import DataTypeEnum
from src.enums.data_enums.tgentrades_enum import TgentradesEnum

logger = logging.getLogger(__name__)


class AbstractReadRawLoader(ABC):
    def _check_is_header(self, series: pd.Series) -> bool:
        return series.str.match(r"^[A-Za-z_]+$").all()

    @abstractmethod
    def _custom_process(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("_custom_process not implemented.")

    def _filter_by_ibex(self, df: pd.DataFrame) -> pd.DataFrame:
        assert (
            CcontractsC2Enum.CONTRACT_CODE.value == TgentradesEnum.CONTRACT_CODE.value
        )
        contract_code_label = CcontractsC2Enum.CONTRACT_CODE.value

        df = df[
            df[contract_code_label].str.startswith(
                tuple(config.data_config.contract_code_config.contracts_prefixes),
                na=False,
            )
        ]

        # Select only futures and options with monthly maturity
        # We can identify them because their number of characters
        df = df[
            df[contract_code_label]
            .str.len()
            .isin(
                [
                    config.data_config.contract_code_config.futures_code_len,
                    config.data_config.contract_code_config.options_code_len,
                ]
            )
        ]

        return df

    # Process paths
    def _get_unique_files(self, file_list: t.List[Path]) -> t.List[Path]:
        unique_files = {}
        for file in file_list:
            name_without_extension = "".join(file.name.split(".")[:-1])
            date = name_without_extension.split("_")[
                -1
            ]  # Assuming the date is the last part of the name
            if file.name not in unique_files:
                unique_files[date] = file
        return list(unique_files.values())

    # Process datetimes
    def _prepare_datetime(self, time_str: str) -> str:
        """
        Possible values for time_str: [
            "01/01/3000 9:00:06",
            "01/01/3000 09:00:06",
            "09:00:00.000000",
            "09:00:00:000",
            "08:00:16.002360"
        ]
        """
        time_str = time_str.split(" ")[-1]
        time_str_split = time_str.split(":")
        time_str_split[0] = (
            f"0{time_str_split[0]}"
            if len(time_str_split[0]) == 1
            else time_str_split[0]
        )

        if len(time_str_split) == 4:
            time_str = f"{time_str_split[0]}:{time_str_split[1]}:{time_str_split[2]}.{time_str_split[3]}"
        elif len(time_str_split) != 3:
            raise ValueError(f"Unexpected time format: {time_str}")
        else:
            time_str = ":".join(time_str_split)

        time_str_split = time_str.split(".")
        if len(time_str_split) == 1:
            time_str = f"{time_str_split[0]}.000000"

        return time_str

    # Convert data types
    def _convert_data_types(
        self,
        df: pd.DataFrame,
        selected_columns_dict: t.Dict[Enum, DataTypeEnum],
    ) -> pd.DataFrame:
        for col_enum, dtype in selected_columns_dict.items():
            col = col_enum.value
            if dtype == DataTypeEnum.DATE:
                df[col] = pd.to_datetime(df[col], format="%Y%m%d").dt.date
            elif dtype == DataTypeEnum.DATETIME:
                df[col] = (
                    df[col]
                    .str.strip()
                    .str.split(" ")
                    .str[-1]
                    .apply(self._prepare_datetime)
                )
                df[col] = pd.to_datetime(df[col], format="%H:%M:%S.%f")
            elif dtype == DataTypeEnum.FLOAT:
                df[col] = pd.to_numeric(df[col].str.replace(",", "."), downcast="float")
            elif dtype == DataTypeEnum.INT:
                df[col] = pd.to_numeric(df[col], downcast="integer")

        return df

    def build_database(
        self,
        columns_list: t.List[Enum],
        selected_columns_dict: t.Dict[Enum, DataTypeEnum],
        file_prefix: str,
    ) -> pd.DataFrame:

        # Data raw
        data_raw = []

        # For every year, we read each contract type
        first_year = config.data_config.read_raw_config.first_year
        last_year = config.data_config.read_raw_config.last_year
        for year in tqdm(range(first_year, last_year + 1)):
            path_year = SOURCE_DATA_DIR_PATH / f"{year}"

            file_list = list(path_year.glob(f"{file_prefix}_*.TXT")) + list(
                path_year.glob(f"{file_prefix}_*.M3")
            )
            unique_file_list = self._get_unique_files(file_list)

            for file in unique_file_list:
                if file.is_file():
                    df = pd.read_csv(
                        file,
                        delimiter=";",
                        header=None,
                        dtype="string",
                    )

                    # Custom process for each case. We can use the default one,
                    # which checks if the first row is a header and removes it if so.
                    df = self._custom_process(df=df)

                    # Columns
                    total_columns = df.shape[1]
                    unknown_names = [
                        f"unknown_{i+1}"
                        for i in range(max(0, total_columns - len(columns_list)))
                    ]
                    column_names = [c.value for c in columns_list] + unknown_names

                    # Assign
                    df.columns = column_names

                    # Select only relevant columns
                    relevant_columns = [c.value for c in selected_columns_dict.keys()]
                    df = df[relevant_columns]

                    # IBX filter
                    df = self._filter_by_ibex(df=df)

                    # Convert data types
                    df = self._convert_data_types(
                        df=df, selected_columns_dict=selected_columns_dict
                    )

                    # Metadata
                    df["Year"] = year
                    df["SourceFile"] = file.name

                    # Join data_raw
                    data_raw.append(df)

        # Concatenate final DataFrame
        data_raw = pd.concat(data_raw, ignore_index=True)

        # Save CSV
        RAW_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
        output_file = RAW_DATA_STEP_DIR_PATH / f"{file_prefix}.csv"
        data_raw.to_csv(output_file, index=False, encoding="utf-8", sep=";")

        logger.info(f"DF (with shape {data_raw.shape}) saved in: {output_file}.")

        return data_raw
