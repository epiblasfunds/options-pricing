"""Callback registration."""

from src.dashboard.dashboard.callbacks.behaviour_surface import (
    register_behaviour_callbacks,
)
from src.dashboard.dashboard.callbacks.diagnosis import (
    register_diagnosis_callbacks,
)
from src.dashboard.dashboard.callbacks.equivalent_models import (
    register_equivalent_callbacks,
)
from src.dashboard.dashboard.callbacks.global_explainability import (
    register_global_callbacks,
)
from src.dashboard.dashboard.callbacks.model_loading import (
    register_model_loading_callbacks,
)
from src.dashboard.dashboard.callbacks.sample_explainability import (
    register_sample_callbacks,
)


def register_callbacks(app, services) -> None:
    """Register all callback groups."""

    register_model_loading_callbacks(app, services)
    register_equivalent_callbacks(app, services)
    register_global_callbacks(app, services)
    register_behaviour_callbacks(app, services)
    register_sample_callbacks(app, services)
    register_diagnosis_callbacks(app, services)

