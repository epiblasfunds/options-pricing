import json


class DashboardRuntimeConfig:
    def _load_config(self, dashboard_models_config_file_path: str) -> None:
        with open(dashboard_models_config_file_path, "r", encoding="utf-8") as file:
            dashboard_models_config = json.load(file)

        runtime_config = dashboard_models_config["dashboard_runtime"]
        self.random_state = runtime_config["random_state"]
        self.surrogate_depths = runtime_config["surrogate_depths"]
        self.surrogate_max_depth = runtime_config["surrogate_max_depth"]
        self.surrogate_min_samples_leaf = runtime_config["surrogate_min_samples_leaf"]
        self.surrogate_sample_size = runtime_config["surrogate_sample_size"]
        self.shap_background_size = runtime_config["shap_background_size"]
        self.shap_explain_size = runtime_config["shap_explain_size"]
        self.shap_permutations = runtime_config["shap_permutations"]
        self.neighbors_sample_size = runtime_config["neighbors_sample_size"]
        self.diagnosis_sample_size = runtime_config["diagnosis_sample_size"]
        self.surface_grid_size = runtime_config["surface_grid_size"]
        self.ice_sample_size = runtime_config["ice_sample_size"]
        self.curve_points = runtime_config["curve_points"]
        self.cache_entries = runtime_config["cache_entries"]

    def __init__(self, dashboard_models_config_file_path: str):
        self._load_config(dashboard_models_config_file_path)
