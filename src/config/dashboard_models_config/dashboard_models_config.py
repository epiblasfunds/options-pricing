import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DashboardBuildConfig:
    surrogate_depths: tuple[int, ...]
    sample_option_size: int
    behaviour_anchor_size: int
    neighbors_k: int


class DashboardModelsConfig:
    def __init__(self, dashboard_models_config_file_path: Path):
        with open(dashboard_models_config_file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        features_config = payload["dashboard_features"]
        settings_config = payload["dashboard_settings"]
        build_config = settings_config["build_config"]

        self.error_metrics = tuple(features_config["error_metrics"])
        self.build_config = DashboardBuildConfig(
            surrogate_depths=tuple(int(depth) for depth in build_config["surrogate_depths"]),
            sample_option_size=int(build_config["sample_option_size"]),
            behaviour_anchor_size=int(build_config["behaviour_anchor_size"]),
            neighbors_k=int(build_config["neighbors_k"]),
        )
        self.random_state = int(settings_config["random_state"])
        self.surrogate_min_samples_leaf = int(settings_config["surrogate_min_samples_leaf"])
        self.surrogate_sample_size = int(settings_config["surrogate_sample_size"])
        self.shap_background_size = int(settings_config["shap_background_size"])
        self.shap_explain_size = int(settings_config["shap_explain_size"])
        self.shap_permutations = int(settings_config["shap_permutations"])
        self.neighbors_sample_size = int(settings_config["neighbors_sample_size"])
        self.diagnosis_sample_size = int(settings_config["diagnosis_sample_size"])
        self.surface_grid_size = int(settings_config["surface_grid_size"])
        self.ice_sample_size = int(settings_config["ice_sample_size"])
        self.curve_points = int(settings_config["curve_points"])
        self.cache_entries = int(settings_config["cache_entries"])
        self.symbolic_sample_size = int(settings_config["symbolic_sample_size"])
        self.symbolic_niterations = int(settings_config["symbolic_niterations"])
        self.symbolic_populations = int(settings_config["symbolic_populations"])
        self.symbolic_population_size = int(
            settings_config["symbolic_population_size"]
        )
        self.symbolic_topn = int(settings_config["symbolic_topn"])
        self.symbolic_ncycles_per_iteration = int(
            settings_config["symbolic_ncycles_per_iteration"]
        )
        self.symbolic_min_candidate_equations = int(
            settings_config["symbolic_min_candidate_equations"]
        )
        self.symbolic_maxsize = int(settings_config["symbolic_maxsize"])
        self.symbolic_maxdepth = int(settings_config["symbolic_maxdepth"])
        self.symbolic_timeout_seconds = int(
            settings_config["symbolic_timeout_seconds"]
        )
