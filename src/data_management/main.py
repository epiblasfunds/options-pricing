from config.config import config

# CCONTRACTS_C2
df = build_data_raw(
    columns_list=config.ccontracts_c2_columns_list,
    selected_columns_dict=config.ccontracts_c2_columns_selected_dict,
    file_prefix=config.tgentrades_prefix,
)

# TGENTRADES
df = build_data_raw(
    columns_list=config.tgentrades_columns_list,
    selected_columns_dict=config.tgentrades_columns_selected_dict,
    file_prefix=config.tgentrades_prefix,
    custom_processing_func=tgentrades_custom_process,
)
