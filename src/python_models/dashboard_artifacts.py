from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


@dataclass
class StoredShapExplanation:
    method: str
    feature_names: list[str]
    index: list[t.Any]
    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray
    display_data: np.ndarray | None
    predictions: np.ndarray
    mean_abs_shap: dict[str, float]

    def select(self, row_index: t.Any) -> "StoredShapExplanation":
        position = self.index.index(row_index)
        return StoredShapExplanation(
            method=self.method,
            feature_names=list(self.feature_names),
            index=[row_index],
            values=np.asarray(self.values[position: position + 1]),
            base_values=np.asarray(self.base_values[position: position + 1]),
            data=np.asarray(self.data[position: position + 1]),
            display_data=(
                None
                if self.display_data is None
                else np.asarray(self.display_data[position: position + 1])
            ),
            predictions=np.asarray(self.predictions[position: position + 1]),
            mean_abs_shap=dict(self.mean_abs_shap),
        )

    def to_explanation(self) -> shap.Explanation:
        return shap.Explanation(
            values=self.values,
            base_values=self.base_values,
            data=self.data,
            display_data=self.display_data,
            feature_names=self.feature_names,
        )


@dataclass
class DiagnosisArtifact:
    metrics: dict[str, float]
    plot_frame: pd.DataFrame
    error_heatmap: pd.DataFrame
    financial_warnings: list[str]


@dataclass
class ManualApiStubResponse:
    prediction: float
    summary: str
    reference_sample_index: t.Any | None = None
