import json


class TrainTestSplitConfig:
    def _load_config(self, data_config_file_path: str):
        with open(data_config_file_path, "r") as f:
            data_config = json.load(f)

        train_test_split_config = data_config["training_data_config"]["train_test_split_config"]

        self.train_size = train_test_split_config["train_size"]
        self.lag = train_test_split_config["lag"]

    def __init__(self, data_config_file_path: str):
        self._load_config(data_config_file_path)
