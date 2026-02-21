import logging
from pathlib import Path

import pandas as pd

from src.config.config import PRODUCT_SPLIT_DATA_STEP_DIR_PATH, config

logger = logging.getLogger(__name__)


def trades_by_contract_type(
    trades_contract_filename: Path,
) -> dict[str, pd.DataFrame]:

    # Read CSV trades with contracts
    df = pd.read_csv(
        trades_contract_filename,
        delimiter=";",
        header=0,
        dtype="string",
    )

    results = {}

    # Get config for contract type
    contract_types_config = config.data_config.product_split_config.contract_types

    for contract_type, cfg in contract_types_config.items():
        filtered_df = df[
            df[
                config.data_config.product_split_config.filter_contract_column
            ].str.startswith(tuple(cfg["prefixes"]), na=False)
        ].copy()
        filtered_df.rename(
            columns={
                config.data_config.product_split_config.filter_contract_column: cfg[
                    "contract_column_new"
                ]
            },
            inplace=True,
        )

        # Select columns
        filtered_df = filtered_df[cfg["columns"]]

        # Store result
        results[contract_type] = filtered_df

        # Save CSV
        PRODUCT_SPLIT_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
        output_filename = f"{contract_type}_{config.data_config.product_split_config.output_filename_contracts}"
        output_file = PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"{output_filename}.csv"
        filtered_df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    logger.info(f"DF (with shape {filtered_df.shape}) saved in: {output_file}.")

    return results
