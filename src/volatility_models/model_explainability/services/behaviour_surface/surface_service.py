"""Surface, slice, ICE, and ALE computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.volatility_models.model_explainability.config import DEFAULT_SETTINGS
from src.volatility_models.model_explainability.services.shared.feature_schema import (
    FeatureSchema,
)
from src.volatility_models.model_explainability.services.shared.prediction_service import (
    PredictionService,
)
from src.volatility_models.model_explainability.utils.feature_utils import (
    apply_feature_override,
)
from src.volatility_models.model_explainability.utils.sampling import (
    quantile_grid,
    sample_frame,
)


class SurfaceService:
    """Generate model-behaviour views around a reference contract."""

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
        surface_size = moneyness_points or DEFAULT_SETTINGS.surface_grid_size
        maturity_size = maturity_points or DEFAULT_SETTINGS.surface_grid_size
        anchor_frame = anchor.to_frame().T
        base_underlying = float(anchor_frame["UnderlyingPrice"].iloc[0])
        moneyness_values = np.linspace(0.8, 1.2, surface_size)
        maturity_values = np.linspace(
            1.0,
            max(float(anchor_frame["TimeToExpiration"].iloc[0]) * 1.5, 30.0),
            maturity_size,
        )
        rows: list[pd.DataFrame] = []
        for maturity in maturity_values:
            for moneyness in moneyness_values:
                row = anchor_frame.copy()
                row["TimeToExpiration"] = maturity
                row["UnderlyingPrice"] = base_underlying
                row["StrikePrice"] = base_underlying / moneyness
                row["Moneyness"] = moneyness
                row["LogMoneyness"] = np.log(moneyness)
                row["AbsLogMoneyness"] = abs(np.log(moneyness))
                rows.append(row)
        grid = pd.concat(rows, ignore_index=True)
        grid["PredictedVolatility"] = self.prediction_service.predict_frame(
            model_id, grid
        ).values
        return grid

    def compute_ice_curves(
        self,
        model_id: str,
        frame: pd.DataFrame,
        feature_name: str,
    ) -> pd.DataFrame:
        sampled = sample_frame(
            frame,
            max_rows=DEFAULT_SETTINGS.ice_sample_size,
            random_state=DEFAULT_SETTINGS.random_state,
        )
        if feature_name in sampled.columns:
            values = quantile_grid(sampled[feature_name], DEFAULT_SETTINGS.curve_points)
        elif feature_name in {"Moneyness", "LogMoneyness"}:
            base_series = (
                sampled["Moneyness"]
                if feature_name == "Moneyness"
                else sampled["LogMoneyness"]
            )
            values = quantile_grid(base_series, DEFAULT_SETTINGS.curve_points)
        else:
            raise KeyError(f"Unsupported ICE feature: {feature_name}")

        rows: list[dict] = []
        for sample_index, (_, sample_row) in enumerate(sampled.iterrows()):
            base_frame = sample_row.to_frame().T
            for value in values:
                adjusted = apply_feature_override(base_frame, feature_name, value)
                prediction = float(
                    self.prediction_service.predict_frame(model_id, adjusted).iloc[0]
                )
                rows.append(
                    {
                        "sample_id": sample_index,
                        "feature_value": value,
                        "prediction": prediction,
                    }
                )
        return pd.DataFrame(rows)

    def compute_ale(
        self,
        model_id: str,
        frame: pd.DataFrame,
        feature_name: str,
        bins: int = 12,
    ) -> pd.DataFrame:
        if feature_name not in frame.columns and feature_name not in {
            "Moneyness",
            "LogMoneyness",
        }:
            raise KeyError(f"Unsupported ALE feature: {feature_name}")

        series = (
            frame[feature_name]
            if feature_name in frame.columns
            else frame["Moneyness" if feature_name == "Moneyness" else "LogMoneyness"]
        )
        edges = (
            pd.Series(series)
            .dropna()
            .quantile(np.linspace(0.05, 0.95, bins + 1))
            .drop_duplicates()
            .tolist()
        )
        if len(edges) < 2:
            return pd.DataFrame(columns=["feature_value", "ale"])

        increments: list[float] = []
        centers: list[float] = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (series >= lower) & (series <= upper)
            bucket = frame.loc[mask]
            if bucket.empty:
                continue
            lower_frame = apply_feature_override(bucket, feature_name, lower)
            upper_frame = apply_feature_override(bucket, feature_name, upper)
            delta = (
                self.prediction_service.predict_frame(model_id, upper_frame)
                - self.prediction_service.predict_frame(model_id, lower_frame)
            ).mean()
            increments.append(float(delta))
            centers.append(float((lower + upper) / 2.0))

        ale = np.cumsum(increments)
        ale = ale - ale.mean()
        return pd.DataFrame({"feature_value": centers, "ale": ale})

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
