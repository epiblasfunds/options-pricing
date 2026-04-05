from src.config.dashboard_models_config.dashboard_features_config import (
    DashboardFeaturesConfig,
)
from src.config.dashboard_models_config.dashboard_runtime_config import (
    DashboardRuntimeConfig,
)


class DashboardModelsConfig:
    def __init__(self, dashboard_models_config_file_path: str):
        self.dashboard_features_config = DashboardFeaturesConfig(
            dashboard_models_config_file_path=dashboard_models_config_file_path
        )
        self.dashboard_runtime_config = DashboardRuntimeConfig(
            dashboard_models_config_file_path=dashboard_models_config_file_path
        )
        self.model_input_features = self.dashboard_features_config.model_input_features
        self.categorical_features = self.dashboard_features_config.categorical_features
        self.numerical_features = self.dashboard_features_config.numerical_features
        self.optional_derived_explainability_features = (
            self.dashboard_features_config.optional_derived_explainability_features
        )
        self.target_column = self.dashboard_features_config.target_column
        self.error_metrics = self.dashboard_features_config.error_metrics
        self.random_state = self.dashboard_runtime_config.random_state
        self.surrogate_depths = self.dashboard_runtime_config.surrogate_depths
        self.surrogate_max_depth = self.dashboard_runtime_config.surrogate_max_depth
        self.surrogate_min_samples_leaf = (
            self.dashboard_runtime_config.surrogate_min_samples_leaf
        )
        self.surrogate_sample_size = self.dashboard_runtime_config.surrogate_sample_size
        self.shap_background_size = self.dashboard_runtime_config.shap_background_size
        self.shap_explain_size = self.dashboard_runtime_config.shap_explain_size
        self.shap_permutations = self.dashboard_runtime_config.shap_permutations
        self.neighbors_sample_size = self.dashboard_runtime_config.neighbors_sample_size
        self.diagnosis_sample_size = self.dashboard_runtime_config.diagnosis_sample_size
        self.surface_grid_size = self.dashboard_runtime_config.surface_grid_size
        self.ice_sample_size = self.dashboard_runtime_config.ice_sample_size
        self.curve_points = self.dashboard_runtime_config.curve_points
        self.cache_entries = self.dashboard_runtime_config.cache_entries
