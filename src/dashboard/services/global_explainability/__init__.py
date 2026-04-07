"""Services for the global explainability dashboard tab."""

from src.dashboard.services.global_explainability.shap_service import (
    ShapExplanationResult,
    ShapService,
)

__all__ = ["ShapExplanationResult", "ShapService"]
