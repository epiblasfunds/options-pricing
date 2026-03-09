import logging
import typing as t
from enum import Enum
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config.config import (
    SOURCE_MARKET_DATA_DIR_PATH,
    SOURCE_RATES_DATA_DIR_PATH,
    config,
)
from src.data_management.builders import (
    CContractsC2Builder,
    RatesBuilder,
    TgentradesBuilder,
)
from src.data_management.utils.data_type_utils import convert_data_types
from src.enums.data_enums import DataTypeEnum

logger = logging.getLogger(__name__)


class ReadRawStepLoader:
    @staticmethod
    def _check_is_header(series: pd.Series) -> bool:
        return series.str.match(r"^[A-Za-z_]+$").all()

    def _process_headers(df: pd.DataFrame, skip_secuencia: bool) -> pd.DataFrame:
        is_header = ReadRawStepLoader._check_is_header(df.iloc[0])
        if is_header:
            if skip_secuencia:
                skip_column_list = [
                    c for c, v in df.iloc[0].items() if v.lower().strip() == "secuencia"
                ]
                df.drop(columns=skip_column_list, inplace=True)
            df = df.iloc[1:, :].reset_index(drop=True)

        return df

    # Process paths
    @staticmethod
    def _get_unique_files(file_list: t.List[Path]) -> t.List[Path]:
        unique_files = {}
        for file in file_list:
            name_without_extension = "".join(file.name.split(".")[:-1])
            date = name_without_extension.split("_")[
                -1
            ]  # Assuming the date is the last part of the name
            if file.name not in unique_files:
                unique_files[date] = file
        return list(unique_files.values())

    @staticmethod
    def _read_raw_databases(
        columns_list: t.List[Enum],
        selected_columns_dict: t.Dict[Enum, DataTypeEnum],
        file_prefix: str,
        skip_secuencia: bool,
    ) -> pd.DataFrame:

        # Data raw
        data_raw = []

        # For every year, we read each contract type
        first_year = config.data_config.read_raw_config.first_year
        last_year = config.data_config.read_raw_config.last_year
        for year in tqdm(range(first_year, last_year + 1)):
            path_year = SOURCE_MARKET_DATA_DIR_PATH / f"{year}"

            file_list = list(path_year.glob(f"{file_prefix}_*.TXT")) + list(
                path_year.glob(f"{file_prefix}_*.M3")
            )
            unique_file_list = ReadRawStepLoader._get_unique_files(file_list)

            for file in unique_file_list:
                if file.is_file():
                    df = pd.read_csv(
                        file,
                        delimiter=";",
                        header=None,
                        dtype="string",
                    )

                    # Checks if the first row is a header and removes it if so.
                    df = ReadRawStepLoader._process_headers(
                        df, skip_secuencia=skip_secuencia
                    )

                    # Columns
                    total_columns = df.shape[1]
                    unknown_names = [
                        f"unknown_{i+1}"
                        for i in range(max(0, total_columns - len(columns_list)))
                    ]
                    column_names = [c for c in columns_list] + unknown_names

                    # Assign
                    df.columns = column_names

                    # Select only relevant columns
                    relevant_columns = [c for c in selected_columns_dict.keys()]
                    df = df[relevant_columns]

                    # Convert data types
                    df = convert_data_types(
                        df=df, selected_columns_dict=selected_columns_dict
                    )

                    # Metadata
                    df["Year"] = year
                    df["SourceFile"] = file.name

                    # Join data_raw
                    data_raw.append(df)

        # Concatenate final DataFrame
        data_raw = pd.concat(data_raw, ignore_index=True)

        return data_raw

    @staticmethod
    def _read() -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Ccontracts C2
        ccontracts_c2_df = ReadRawStepLoader._read_raw_databases(
            columns_list=config.data_config.read_raw_config.ccontracts_c2_columns_list,
            selected_columns_dict=config.data_config.read_raw_config.ccontracts_c2_columns_selected_dict,
            file_prefix=config.data_config.read_raw_config.cconctracts_c2_prefix,
            skip_secuencia=False,
        )

        # Tgentrades
        tgentrades_df = ReadRawStepLoader._read_raw_databases(
            columns_list=config.data_config.read_raw_config.tgentrades_columns_list,
            selected_columns_dict=config.data_config.read_raw_config.tgentrades_columns_selected_dict,
            file_prefix=config.data_config.read_raw_config.tgentrades_prefix,
            skip_secuencia=True,
        )

        # EONIA
        eonia_df = pd.read_csv(
            SOURCE_RATES_DATA_DIR_PATH / "ECB_EONIA.csv",
            delimiter=",",
            header=0,
            dtype="string",
        )

        # STR
        str_df = pd.read_csv(
            SOURCE_RATES_DATA_DIR_PATH / "ECB_STR.csv",
            delimiter=",",
            header=0,
            dtype="string",
        )

        return ccontracts_c2_df, tgentrades_df, eonia_df, str_df

    @staticmethod
    def load() -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        # Read raw databases
        ccontracts_c2_df, tgentrades_df, eonia_df, str_df = ReadRawStepLoader._read()

        # Builders
        ccontracts_c2_df = CContractsC2Builder.build(ccontracts_c2_df)
        tgentrades_df = TgentradesBuilder.build(tgentrades_df)
        rates_df = RatesBuilder.build(eonia_df, str_df)

        return ccontracts_c2_df, tgentrades_df, rates_df
