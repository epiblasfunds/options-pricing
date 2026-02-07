import typing as t
from enum import Enum
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# Data types for columns
class DataType(Enum):
    DATE = 0
    DATETIME = 1
    TEXT = 2
    FLOAT = 3
    INT = 4

# CCONTRACTS_C2 types
class CONTRACT_TYPE(Enum):
    FUTURE = "FUTURE"
    OPTION = "OPTION"

# Years to process
FIRST_YEAR = 2017
LAST_YEAR = 2022

# Custom process
def check_is_header(series: pd.Series) -> bool:
    return series.str.match(r"^[A-Za-z_]+$").all()

def default_custom_process(df: pd.DataFrame) -> pd.DataFrame:
    is_header = check_is_header(df.iloc[0])
    if is_header:
        df = df.iloc[1:, :].reset_index(drop=True)
        
    return df

def tgentrades_custom_process(df: pd.DataFrame) -> pd.DataFrame:
    is_header = check_is_header(df.iloc[0])
    if is_header:
        skip_column_list = [c for c, v in df.iloc[0].items() if v.lower().strip() == "secuencia"]
        df.drop(columns=skip_column_list, inplace=True)
        df = df.iloc[1:, :].reset_index(drop=True)

    return df

# Process paths
def get_unique_files(file_list:t.List[Path]) -> t.List[Path]:
    unique_files = {}
    for file in file_list:
        name_without_extension = "".join(file.name.split(".")[:-1])
        date = name_without_extension.split("_")[-1]  # Assuming the date is the last part of the name
        if file.name not in unique_files:
            unique_files[date] = file
    return list(unique_files.values())

# Process datetimes
def prepare_datetime(time_str: str) -> str:
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
    time_str_split[0] = f"0{time_str_split[0]}" if len(time_str_split[0]) == 1 else time_str_split[0]

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

# Main function to build raw data
def build_data_raw(
        columns_list:t.List[str],
        selected_columns_dict:t.Dict[str, DataType],
        file_prefix: str,
        custom_processing_func: t.Optional[t.Callable] = default_custom_process
    ) -> pd.DataFrame:

    # Data raw
    data_raw = []

    # For every year, we read each contract type
    for year in tqdm(range(FIRST_YEAR, LAST_YEAR + 1)):
        path_year = Path(f"data/{year}")

        file_list = list(path_year.glob(f"{file_prefix}_*.TXT")) + list(path_year.glob(f"{file_prefix}_*.M3"))
        unique_file_list = get_unique_files(file_list)

        for file in unique_file_list:
            if file.is_file():
                df = pd.read_csv(
                    file,
                    delimiter=";",
                    header=None,
                    dtype="string",
                )

                # Custom process for each case. We can use the default one, which checks if the first row is a header and removes it if so.
                df = custom_processing_func(df=df)

                # Columns
                total_columns = df.shape[1]
                unknown_names = [
                    f"unknown_{i+1}"
                    for i in range(max(0, total_columns - len(columns_list)))
                ]
                column_names = columns_list + unknown_names

                # Assign
                df.columns = column_names

                # Select only relevant columns
                df = df[list(selected_columns_dict.keys())]

                # IBX mask
                IBX_mask = df["ContractCode"].str.contains(
                    ("IBX"),
                    na=False
                )
                df = df[IBX_mask]

                # Convert data types
                for col, dtype in selected_columns_dict.items():
                    if dtype == DataType.DATE:
                        df[col] = pd.to_datetime(df[col], format="%Y%m%d").dt.date    
                    elif dtype == DataType.DATETIME:
                        
                        try:
                            df[col] = df[col].str.strip().str.split(" ").str[-1].apply(prepare_datetime)
                            df[col] = pd.to_datetime(df[col], format="%H:%M:%S.%f")
                        except Exception as e:
                            raise ValueError(f"Unexpected format in column {col}. file: {file.name} Error: {e}")
                    elif dtype == DataType.FLOAT:
                        df[col] = pd.to_numeric(df[col].str.replace(",", "."), downcast="float")
                    elif dtype == DataType.INT:
                        df[col] = pd.to_numeric(df[col], downcast="integer")
                
                # Metadata
                df["Year"] = year
                df["SourceFile"] = file.name
                
                # Join data_raw
                data_raw.append(df)

    # Concatenate final DataFrame
    data_raw = pd.concat(data_raw, ignore_index=True)

    # Save CSV
    output_dir = Path(f"raw_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{file_prefix}.csv"
    data_raw.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    print(f"\nArchivo guardado en: {output_file}")
    print(f"Total filas finales: {len(data_raw)}")

    return data_raw


def merge_trade_with_contracts(
        trades_filename: str,
        contracts_filename: str,
        merge_columns: t.List[str],
        selected_columns_dict: t.Dict[str, DataType]
) -> pd.DataFrame:
    
    # Read CSVs
    trades_df = pd.read_csv(
        Path(f"raw_data/{trades_filename}"),
        delimiter=";",
        header=0,
        dtype="string",
    )
    contracts_df = pd.read_csv(
        Path(f"raw_data/{contracts_filename}"),
        delimiter=";",
        header=0,
        dtype="string",
    )

    # Merge
    merged_df = trades_df.merge(
        contracts_df,
        on = merge_columns,
        how = "left",
        suffixes=("", "_contract")
    )

    # Select only relevant columns
    merged_df = merged_df[list(selected_columns_dict.keys())]

    # Save CSV
    output_dir = Path(f"raw_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"TRADE_IBEX_DATABASE.csv"
    merged_df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    print(f"\nArchivo guardado en: {output_file}")
    print(f"Total filas finales: {len(merged_df)}")

    return merged_df

def filtered_contracts_with_trades(
        contract_type: str
) -> pd.DataFrame:
    
    # Read CSV trades with contracts
    file = Path("raw_data/TRADE_IBEX_DATABASE.csv")
    df = pd.read_csv(
        file,
        delimiter=";",
        header=0,
        dtype="string",
    )

    # Filter by contract type
    if contract_type == CONTRACT_TYPE.FUTURE.value:
        prefixes = ("FIBX",)
        contract_column_new = "FutureContractCode"
        filename_prefix = "FUTURE"
    elif contract_type == CONTRACT_TYPE.OPTION.value:
        prefixes = ("CIBX", "PIBX")
        contract_column_new = "OptionContractCode"
        filename_prefix = "OPTIONS"
    else:
        raise ValueError(f"Tipo de contrato desconocido: {contract_type}")

    # Apply filter
    df = df[df['ContractCode'].str.startswith(prefixes, na=False)]

    # Change column names
    df.rename(columns={"ContractCode": contract_column_new}, inplace=True)

    # Save CSV
    output_dir = Path(f"raw_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{filename_prefix}_TRADE_IBEX_DATABASE.csv"
    df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    print(f"\nArchivo guardado en: {output_file}")
    print(f"Total filas finales: {len(df)}")

    return df

def build_options_underlying_ibex_database():
    # Read CSVs
    options_trade_ibex_file = Path("raw_data/OPTIONS_TRADE_IBEX_DATABASE.csv")
    future_trade_ibex_file = Path("raw_data/FUTURE_TRADE_IBEX_DATABASE.csv")

    # Load dataframes
    options_df = pd.read_csv(
        options_trade_ibex_file,
        delimiter=";",
        header=0,
        dtype="string",
    )
    future_df = pd.read_csv(
        future_trade_ibex_file,
        delimiter=";",
        header=0,
        dtype="string",
    )

    # Select relevant columns
    options_df = options_df[["OptionContractCode", "MaturityDate"]]


    # Drop duplicates in future_df to avoid multiple matches when merging with options_df
    future_maturity_df = future_df[["FutureContractCode", "MaturityDate"]].drop_duplicates(subset=["MaturityDate"])

    # Merge
    df = options_df.merge(
        future_maturity_df,
        how="left",
        on="MaturityDate",
    )
    # Select relevant columns
    df = df[["OptionContractCode", "FutureContractCode", "MaturityDate"]]

    # Save CSV
    output_file = Path("raw_data/OPTIONS_UNDERLYING_IBEX_DATABASE.csv")
    df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    print(f"\nArchivo guardado en: {output_file}")
    print(f"Total filas finales: {len(df)}")

    return df