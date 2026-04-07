"""Surface, slice, ICE, and ALE access backed by dashboard artifacts."""

from __future__ import annotations

import pandas as pd

from src.dashboard.services.shared.feature_schema import FeatureSchema
from src.dashboard.services.shared.prediction_service import PredictionService


class SurfaceService:
    """Serve precomputed behaviour and surface views from dashboard bundles."""

    def __init__(
        self,
        prediction_service: PredictionService,
        feature_schema: FeatureSchema,
    ) -> None:
        self.prediction_service = prediction_service
        self.feature_schema = feature_schema

    def default_anchor(self, frame: pd.DataFrame) -> pd.Series:
        defaults = self.feature_schema.defaults_from_frame(frame, raw_only=True)
        return pd.Series(defaults)

    def build_surface(
        self,
        model_id: str,
        anchor: pd.Series,
        moneyness_points: int | None = None,
        maturity_points: int | None = None,
    ) -> pd.DataFrame:
        bundle = self.prediction_service.load_bundle(model_id)
        anchor_index = (
            anchor.name
            if anchor.name in bundle.dashboard_model.behaviour_anchor_indices
            else (
                bundle.dashboard_model.behaviour_anchor_indices[0]
                if bundle.dashboard_model.behaviour_anchor_indices
                else None
            )
        )
        if anchor_index is None:
            raise KeyError(
                "No precomputed behaviour anchors were exported for this model."
            )
        return bundle.dashboard_model.surface_for_anchor(anchor_index)

    def compute_ice_curves(
        self,
        model_id: str,
        frame: pd.DataFrame,
        feature_name: str,
    ) -> pd.DataFrame:
        bundle = self.prediction_service.load_bundle(model_id)
        cached = bundle.dashboard_model.ice_for_feature(feature_name)
        if cached.empty:
            raise KeyError(
                f"No precomputed ICE curves were exported for feature: {feature_name}"
            )
        return cached

    def compute_ale(
        self,
        model_id: str,
        frame: pd.DataFrame,
        feature_name: str,
        bins: int = 12,
    ) -> pd.DataFrame:
        bundle = self.prediction_service.load_bundle(model_id)
        cached = bundle.dashboard_model.ale_for_feature(feature_name)
        if cached.empty:
            raise KeyError(
                f"No precomputed ALE data were exported for feature: {feature_name}"
            )
        return cached

    def financial_checks(self, surface_frame: pd.DataFrame) -> list[str]:
        warnings: list[str] = []
        if surface_frame.empty:
            return ["No local surface could be generated for the financial checks."]
        pivot = surface_frame.pivot_table(
            index="TimeToExpiration",
            columns="Moneyness",
            values="PredictedVolatility",
        ).sort_index()
        smile_diff = pivot.diff(axis=1).abs().max().max()
        term_diff = pivot.diff(axis=0).abs().max().max()
        if pd.notna(smile_diff) and smile_diff > 0.20:
            warnings.append(
                "Heuristic warning: adjacent smile points show large volatility jumps."
            )
        if pd.notna(term_diff) and term_diff > 0.20:
            warnings.append(
                "Heuristic warning: adjacent maturity points show large term-structure jumps."
            )
        if not warnings:
            warnings.append(
                "No large discontinuities were detected by the heuristic checks."
            )
        return warnings
