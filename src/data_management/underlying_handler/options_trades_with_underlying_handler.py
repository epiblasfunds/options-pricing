from pathlib import Path

import pandas as pd

from src.config.config import UNDERLYING_DATA_STEP_DIR_PATH, config
from src.enums.contract_type_enum import ContractTypeEnum
from src.enums.futures_trade_ibex_database_enum import FuturesTradeIbexDatabaseEnum
from src.enums.options_trade_ibex_database_enum import OptionsTradeIbexDatabaseEnum


def build_options_trade_underlying(
        options_trades_filename: Path,
        futures_trades_filename: Path,
        options_underlying_filename: Path,
) -> pd.DataFrame:
    
    # Read CSV options trades
    df_options = pd.read_csv(
        options_trades_filename,
        delimiter=";",
        header=0,
        dtype="string",
        )

    # Read CSV futures trades
    df_futures = pd.read_csv(
        futures_trades_filename,
        delimiter=";",
        header=0,
        dtype="string",
        )
    
    # Read CSV options & underlying
    df_options_underlying = pd.read_csv(
        options_underlying_filename,
        delimiter=";",
        header=0,
        dtype="string",
    )

    # Create exec_datetime (SessionDate + ExecTime)
    exec_datetime_column_name = "ExecDateTime"
    for df in (df_options, df_futures):
        if df is df_options:
            exec_time_column = OptionsTradeIbexDatabaseEnum.EXEC_TIME.value
            session_date_column = OptionsTradeIbexDatabaseEnum.SESSION_DATE.value
        elif df is df_futures:
            exec_time_column = FuturesTradeIbexDatabaseEnum.EXEC_TIME.value
            session_date_column = FuturesTradeIbexDatabaseEnum.SESSION_DATE.value
        
        df[exec_time_column] = df[exec_time_column].astype(str).str.split().str[-1]
        
        # Ensure microseconds are present by appending ".000000" when missing
        df[exec_time_column] = df[exec_time_column].apply(
            lambda x: x if "." in x else x + ".000000"
        )

        df[exec_datetime_column_name] = pd.to_datetime(
            df[session_date_column].astype(str) + " " + df[exec_time_column],
            format="%Y-%m-%d %H:%M:%S.%f",
        )
        
    # Rename exec_datetime of future
    new_column_exec_datetime_futures_name = f"{exec_datetime_column_name}_{ContractTypeEnum.FUTURES.name}"
    df_futures = df_futures.rename(columns={exec_datetime_column_name: new_column_exec_datetime_futures_name})

    # Join option with its underlying future
    df_options = df_options.merge(
        df_options_underlying,
        on=OptionsTradeIbexDatabaseEnum.OPTION_CONTRACT_CODE.value,
        how="left"
    )

    # Order by exec_datetime
    df_options = df_options.sort_values(exec_datetime_column_name).reset_index(drop=True)
    df_futures = df_futures.sort_values(new_column_exec_datetime_futures_name).reset_index(drop=True)

    # As-of join: Last trade of the underlying FUTURE with exec_datetime <= exec_datetime of the option
    df = pd.merge_asof(
        df_options,
        df_futures,
        by=FuturesTradeIbexDatabaseEnum.FUTURE_CONTRACT_CODE.value,
        left_on=exec_datetime_column_name,
        right_on=new_column_exec_datetime_futures_name,
        direction="backward",
        suffixes=("", "_future")
    )

    # Rename columns
    df = df.rename(columns={
        OptionsTradeIbexDatabaseEnum.TRADE_PRICE.value: "TradePriceOption",
        f"{FuturesTradeIbexDatabaseEnum.TRADE_PRICE.value}_future": "UnderelayingPrice",
        new_column_exec_datetime_futures_name: "UnderelayingExecTime"
    })

    # Select final columns
    print(df)
    print(df.columns)
    
    df = df[config.data_config.underlying_config.options_trade_underlying_ibex_database_columns]
    
    # Save CSV
    UNDERLYING_DATA_STEP_DIR_PATH.mkdir(parents=True, exist_ok=True)
    output_filename = config.data_config.underlying_config.output_filename
    output_file = UNDERLYING_DATA_STEP_DIR_PATH / f"{output_filename}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8", sep=";")

    print(f"\nArchivo guardado en: {output_file}")
    print(f"Total filas finales: {len(df)}")

    return df
     