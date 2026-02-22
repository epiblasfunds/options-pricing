import logging
import typing as t
from enum import Enum
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config.config import SOURCE_DATA_DIR_PATH, config
from src.data_management.builders import CContractsC2Builder, TgentradesBuilder
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

    # Process datetimes
    @staticmethod
    def _prepare_datetime(time_str: str) -> str:
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
    @staticmethod
    def _convert_data_types(
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
                    .apply(ReadRawStepLoader._prepare_datetime)
                )
                df[col] = pd.to_datetime(df[col], format="%H:%M:%S.%f")
            elif dtype == DataTypeEnum.FLOAT:
                df[col] = pd.to_numeric(df[col].str.replace(",", "."), downcast="float")
            elif dtype == DataTypeEnum.INT:
                df[col] = pd.to_numeric(df[col], downcast="integer")

        return df

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
            path_year = SOURCE_DATA_DIR_PATH / f"{year}"

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
                    column_names = [c.value for c in columns_list] + unknown_names

                    # Assign
                    df.columns = column_names

                    # Select only relevant columns
                    relevant_columns = [c.value for c in selected_columns_dict.keys()]
                    df = df[relevant_columns]

                    # Convert data types
                    df = ReadRawStepLoader._convert_data_types(
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
    def load() -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        # Ccontracts C2
        ccontracts_c2_df = ReadRawStepLoader._read_raw_databases(
            columns_list=config.data_config.read_raw_config.ccontracts_c2_columns_list,
            selected_columns_dict=config.data_config.read_raw_config.ccontracts_c2_columns_selected_dict,
            file_prefix=config.data_config.read_raw_config.cconctracts_c2_prefix,
            skip_secuencia=False,
        )
        ccontracts_c2_df = CContractsC2Builder.build(ccontracts_c2_df)

        # Tgentrades
        tgentrades_df = ReadRawStepLoader._read_raw_databases(
            columns_list=config.data_config.read_raw_config.tgentrades_columns_list,
            selected_columns_dict=config.data_config.read_raw_config.tgentrades_columns_selected_dict,
            file_prefix=config.data_config.read_raw_config.tgentrades_prefix,
            skip_secuencia=True,
        )
        tgentrades_df = TgentradesBuilder.build(tgentrades_df)

        return ccontracts_c2_df, tgentrades_df
