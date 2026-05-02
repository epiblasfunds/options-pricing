"""Services for the global explainability dashboard tab."""

from src.dashboard.services.global_explainability.shap_service import (
    AUXILIARY_FEATURE_LABEL,
    FULL_FEATURE_SCOPE,
    MAIN_FEATURE_SCOPE,
    ShapExplanationResult,
    ShapService,
)

__all__ = [
    "AUXILIARY_FEATURE_LABEL",
    "FULL_FEATURE_SCOPE",
    "MAIN_FEATURE_SCOPE",
    "ShapExplanationResult",
    "ShapService",
]
