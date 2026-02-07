import typing as t
from pathlib import Path

import pandas as pd

from src.config.config import MERGE_RAW_DATA_STEP_DIR_PATH, config


def merge_trade_with_contracts(
        trades_filename: str,
        contracts_filename: str,
        merge_columns: t.List[str],
        selected_columns_list: t.List[str],
) -> pd.DataFrame:
    
    # Read CSVs
    trades_df = pd.read_csv(
        Path(trades_filename),
        delimiter=";",
        header=0,
        dtype="string",
    )
    contracts_df = pd.read_csv(
        Path(contracts_filename),
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
    merged_df = merged_df[selected_columns_list]

    # Save CSV
    MERGE_RAW_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    output_filename = config.data_config.merge_raw_config.output_filename
    output_file = MERGE_RAW_DATA_STEP_DIR_PATH / f"{output_filename}.csv"
    merged_df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    print(f"\nArchivo guardado en: {output_file}")
    print(f"Total filas finales: {len(merged_df)}")

    return merged_df
