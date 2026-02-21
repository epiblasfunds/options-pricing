import json


class ContractCodeConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        contract_code_config = data_config["contract_code"]

        self.futures_code_len = contract_code_config["futures_code_len"]
        self.options_code_len = contract_code_config["options_code_len"]
        self.contracts_prefixes = contract_code_config["contracts_prefixes"]

        # cheatsheet
        self.futures_code_month = contract_code_config["futures_code_month"]
        self.call_code_month = contract_code_config["call_code_month"]
        self.put_code_month = contract_code_config["put_code_month"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
