"""Registry-driven metric system."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricDefinition:
    """Definition of one metric."""

    name: str
    label: str
    function: Callable[[pd.Series, pd.Series], float]
    higher_is_better: bool
    formatter: Callable[[float], str]
    description: str | None = None

    def compute(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return float(self.function(y_true, y_pred))

    def format(self, value: float) -> str:
        return self.formatter(value)


class MetricsRegistry:
    """Registry for configurable error metrics."""

    def __init__(self) -> None:
        self._definitions: dict[str, MetricDefinition] = {}

    def register(self, definition: MetricDefinition) -> None:
        self._definitions[definition.name] = definition

    def get(self, name: str) -> MetricDefinition:
        return self._definitions[name]

    def resolve(self, names: list[str] | tuple[str, ...]) -> list[MetricDefinition]:
        return [self.get(name) for name in names]

    def compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        metric_names: list[str] | tuple[str, ...],
    ) -> dict[str, float]:
        aligned = pd.DataFrame(
            {
                "y_true": pd.Series(y_true).astype(float).reset_index(drop=True),
                "y_pred": pd.Series(y_pred).astype(float).reset_index(drop=True),
            }
        )
        finite_mask = np.isfinite(aligned["y_true"]) & np.isfinite(aligned["y_pred"])
        aligned = aligned.loc[finite_mask].reset_index(drop=True)
        aligned_true = aligned["y_true"]
        aligned_pred = aligned["y_pred"]
        if aligned.empty:
            return {
                definition.name: float("nan")
                for definition in self.resolve(metric_names)
            }
        return {
            definition.name: definition.compute(aligned_true, aligned_pred)
            for definition in self.resolve(metric_names)
        }

    def format_metric(self, metric_name: str, value: float) -> str:
        return self.get(metric_name).format(value)
