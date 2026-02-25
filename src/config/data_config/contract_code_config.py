import json


class ContractCodeConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        contract_code_config = data_config["contract_code"]

        self.futures_code_len = contract_code_config["futures_code_len"]
        self.options_code_len = contract_code_config["options_code_len"]
        self.contracts_prefixes = contract_code_config["contracts_prefixes"]
        self.strike_starts = contract_code_config["strike_starts"]
        self.strike_ends = contract_code_config["strike_ends"]

        self.month_options_code_idx = contract_code_config["month_options_code_idx"]
        self.year_options_code_idx = contract_code_config["year_options_code_idx"]
        self.month_futures_code_idx = contract_code_config["month_futures_code_idx"]
        self.year_futures_code_idx = contract_code_config["year_futures_code_idx"]

        # cheatsheet
        self.futures_code_month = contract_code_config["futures_code_month"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
