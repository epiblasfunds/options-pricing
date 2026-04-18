import json


class KFoldsConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        kfolds_config = data_config["training_data_config"]["kfolds_config"]

        self.n_folds = kfolds_config["n_folds"]
        self.extra_blocks = kfolds_config["extra_blocks"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
