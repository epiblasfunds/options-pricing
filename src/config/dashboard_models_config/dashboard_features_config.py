import json


class DashboardFeaturesConfig:
    def _load_config(self, dashboard_models_config_file_path: str) -> None:
        with open(dashboard_models_config_file_path, "r", encoding="utf-8") as file:
            dashboard_models_config = json.load(file)

        features_config = dashboard_models_config["dashboard_models"]
        self.model_input_features = features_config["model_input_features"]
        self.categorical_features = features_config["categorical_features"]
        self.numerical_features = features_config["numerical_features"]
        self.optional_derived_explainability_features = features_config[
            "optional_derived_explainability_features"
        ]
        self.target_column = features_config["target_column"]
        self.error_metrics = features_config["error_metrics"]

    def __init__(self, dashboard_models_config_file_path: str):
        self._load_config(dashboard_models_config_file_path)
