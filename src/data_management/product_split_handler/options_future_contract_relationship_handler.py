from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config


def options_future_contract_relationship(
        options_trades_filename: Path,
        futures_trades_filename: Path
) -> pd.DataFrame:
    
    # Read CSV options trades
    try:
        df_options = pd.read_csv(
            options_trades_filename,
            delimiter=";",
            header=0,
            dtype="string",
            )
    except Exception as e:
        raise ValueError(f"Error al leer OPTIONS ({options_trades_filename}): {e}")

    # Read CSV futures trades
    try:
        df_futures = pd.read_csv(
            futures_trades_filename,
            delimiter=";",
            header=0,
            dtype="string",
            )
    except Exception as e:
        raise ValueError(f"Error al leer FUTURES ({futures_trades_filename}): {e}")
    
    # Select relevant columns
    options_df = (
    df_options[["OptionContractCode", "MaturityDate"]]
        .copy()
   )

    futures_df = (
        df_futures[["FutureContractCode", "MaturityDate"]]
        .copy()
    )

    # Merge on MaturityDate
    df = options_df.merge(
        futures_df,
        how="left",
        on="MaturityDate",
    )

    # Save CSV
    PRODUCT_SPLIT_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    output_filename = config.data_config.product_split_config.output_filename_relationship
    output_file = PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    print(f"\nArchivo guardado en: {output_file}")
    print(f"Total filas finales: {len(df)}")

    return df
