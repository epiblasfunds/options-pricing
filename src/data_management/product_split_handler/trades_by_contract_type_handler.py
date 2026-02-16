from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config


def trades_by_contract_type(
        trades_contract_filename: Path,
) -> dict[str, pd.DataFrame]:
    
    # Read CSV trades with contracts
    try:
        df = pd.read_csv(
            trades_contract_filename,
            delimiter=";",
            header=0,
            dtype="string",
            )
    except Exception as e:
        raise ValueError(f"Error al leer el archivo {trades_contract_filename}: {e}")
    
    results = {}
    
    # Get config for contract type
    contract_types_config = config.data_config.product_split_config.contract_types

    for contract_type, cfg in contract_types_config.items():
        filtered_df = df[df["ContractCode"].str.startswith(tuple(cfg["prefixes"]), na=False)].copy()
        filtered_df.rename(columns={"ContractCode": cfg["contract_column_new"]}, inplace=True)

        # Store result
        results[contract_type] = filtered_df

        # Save CSV
        PRODUCT_SPLIT_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
        output_filename = f"{contract_type}_{config.data_config.product_split_config.output_filename_contracts}"
        output_file = PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"
        filtered_df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

        print(f"\nArchivo guardado en: {output_file}")
        print(f"Total filas finales: {len(filtered_df)}")

    return results
