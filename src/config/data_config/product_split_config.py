import json
import typing as t


class ProductSplitConfig:
    def _load_config(self, data_config_file_path: str):
        with(open(data_config_file_path, "r") as f):
            data_config = json.load(f)
        product_split_step_name = "product_split_step"
        self.contract_types = data_config[product_split_step_name]["contract_types"]
        self.output_filename = data_config[product_split_step_name]["output_filename"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
