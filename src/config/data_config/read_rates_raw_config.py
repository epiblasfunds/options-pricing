import json
import typing as t


class ReadRatesRawConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        read_rates_step_name = "read_rates_raw_step"
        rates_config = data_config[read_rates_step_name]

        # Load config values into class attributes
        self.eonia_prefix = rates_config["EONIA_prefix"]
        self.str_prefix = rates_config["STR_prefix"]
        self.euribor3m_prefix = rates_config["EURIBOR3M_prefix"]
        self.euribor6m_prefix = rates_config["EURIBOR6M_prefix"]
        self.euribor12m_prefix = rates_config["EURIBOR12M_prefix"]
        self.idx_date_column = rates_config["idx_date_column"]
        self.idx_rate_column = rates_config["idx_rate_column"]
        self.index_rates_column_name = rates_config["index_rates_column_name"]
        self.unified_overnight_rate_column_name = rates_config["unified_overnight_rate_column_name"]
        self.spread_str_eonia = rates_config["spread_str_eonia"]
        self.cutoff_date_str_eonia = rates_config["cutoff_date_str_eonia"]
        self.free_risk_rates_columns = rates_config["free_risk_rates_columns"]
        self.output_filename = rates_config["output_filename"]

    def get_rates_prefixes(self) -> t.Dict[str, str]:
        return {
            "EONIA": self.eonia_prefix,
            "STR": self.str_prefix,
            "EURIBOR3M": self.euribor3m_prefix,
            "EURIBOR6M": self.euribor6m_prefix,
            "EURIBOR12M": self.euribor12m_prefix,
        }

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
