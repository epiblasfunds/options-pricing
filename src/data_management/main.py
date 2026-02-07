from config.config import config

# CCONTRACTS_C2
df = build_data_raw(
    columns_list=config.data_config.raw_data_config.ccontracts_c2_columns_list,
    selected_columns_dict=config.data_config.raw_data_config.ccontracts_c2_columns_selected_dict,
    file_prefix=config.data_config.raw_data_config.tgentrades_prefix,
)

# TGENTRADES
df = build_data_raw(
    columns_list=config.data_config.raw_data_config.tgentrades_columns_list,
    selected_columns_dict=config.data_config.raw_data_config.tgentrades_columns_selected_dict,
    file_prefix=config.data_config.raw_data_config.tgentrades_prefix,
    custom_processing_func=tgentrades_custom_process,
)
