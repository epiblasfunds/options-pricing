from config.config import config
from src.data_management.read_raw_handler.ccontracts_c2_handler import (
    CContractsC2Handler,
)
from src.data_management.read_raw_handler.tgentrades_handler import TgentradesHandler

# CCONTRACTS_C2
df = CContractsC2Handler().build_database(
    columns_list=config.data_config.read_raw_config.ccontracts_c2_columns_list,
    selected_columns_dict=config.data_config.read_raw_config.ccontracts_c2_columns_selected_dict,
    file_prefix=config.data_config.read_raw_config.cconctracts_c2_prefix,
    contracts_prefixes=config.data_config.read_raw_config.contracts_prefixes,
)
# TGENTRADES
df = TgentradesHandler().build_database(
    columns_list=config.data_config.read_raw_config.tgentrades_columns_list,
    selected_columns_dict=config.data_config.read_raw_config.tgentrades_columns_selected_dict,
    file_prefix=config.data_config.read_raw_config.tgentrades_prefix,
    contracts_prefixes=config.data_config.read_raw_config.contracts_prefixes,
)

# TRADE_IBEX_DATABASE.csv
trades_filename = RAW_DATA_STEP_DIR_PATH / f"{config.data_config.read_raw_config.tgentrades_prefix}.csv"
contracts_filename = RAW_DATA_STEP_DIR_PATH / f"{config.data_config.read_raw_config.cconctracts_c2_prefix}.csv"

# TRADE_IBEX_DATABASE
df = merge_trade_with_contracts(
    trades_filename=trades_filename,
    contracts_filename=contracts_filename,
    merge_columns=config.data_config.merge_raw_config.merge_columns_list,
    selected_columns_list=config.data_config.merge_raw_config.trade_ibex_columns_list,
)


# OPTIONS_TRADE_IBEX_DATABASE & FUTURES_TRADE_IBEX_DATABASE
trades_with_contracts_filename = MERGE_RAW_DATA_STEP_DIR_PATH / f"{config.data_config.merge_raw_config.output_filename}.csv"

dfs = trades_by_contract_type(
    trades_contract_filename=trades_with_contracts_filename,
    )

# OPTIONS_UNDERLYING_IBEX_DATABASE
output_contracts = config.data_config.product_split_config.output_filename_contracts

options_tr_filename = PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"OPTIONS_{output_contracts}.csv"
futures_tr_filename = PRODUCT_SPLIT_DATA_STEP_DIR_PATH / f"FUTURES_{output_contracts}.csv"

df = options_future_contract_relationship(
        options_trades_filename=options_tr_filename,
        futures_trades_filename=futures_tr_filename,
    )
