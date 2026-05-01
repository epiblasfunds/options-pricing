import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sympy


@dataclass
class SymbolicRegressorModel:
    equation: str
    sympy_expression: str
    latex_expression: str
    interpretation: str
    feature_names: list[str]
    used_feature_names: list[str]
    complexity: int
    model_selection: str
    metrics: dict[str, float] = field(default_factory=dict)
    candidate_equations: pd.DataFrame = field(default_factory=pd.DataFrame)
    fidelity_frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    @staticmethod
    def _get_attrs_path(path: Path) -> Path:
        return path / "attributes.json"

    @staticmethod
    def _get_candidates_path(path: Path) -> Path:
        return path / "candidate_equations.csv"

    @staticmethod
    def _get_fidelity_path(path: Path) -> Path:
        return path / "fidelity_frame.csv"

    @classmethod
    def load(cls, path: Path) -> "SymbolicRegressorModel":
        attrs = json.loads(cls._get_attrs_path(path).read_text(encoding="utf-8"))
        candidate_path = cls._get_candidates_path(path)
        fidelity_path = cls._get_fidelity_path(path)
        candidate_equations = (
            pd.read_csv(candidate_path, index_col=0)
            if candidate_path.exists()
            else pd.DataFrame()
        )
        fidelity_frame = (
            pd.read_csv(fidelity_path, index_col=0)
            if fidelity_path.exists()
            else pd.DataFrame()
        )
        return cls(
            equation=attrs["equation"],
            sympy_expression=attrs["sympy_expression"],
            latex_expression=attrs["latex_expression"],
            interpretation=attrs["interpretation"],
            feature_names=list(attrs.get("feature_names", [])),
            used_feature_names=list(attrs.get("used_feature_names", [])),
            complexity=int(attrs["complexity"]),
            model_selection=attrs["model_selection"],
            metrics={
                name: float(value) for name, value in attrs.get("metrics", {}).items()
            },
            candidate_equations=candidate_equations,
            fidelity_frame=fidelity_frame,
        )

    def save(self, path: Path) -> None:
        attrs = {
            "equation": self.equation,
            "sympy_expression": self.sympy_expression,
            "latex_expression": self.latex_expression,
            "interpretation": self.interpretation,
            "feature_names": list(self.feature_names),
            "used_feature_names": list(self.used_feature_names),
            "complexity": int(self.complexity),
            "model_selection": self.model_selection,
            "metrics": {name: float(value) for name, value in self.metrics.items()},
        }
        path.mkdir(parents=True, exist_ok=True)
        self._get_attrs_path(path).write_text(
            json.dumps(attrs, indent=2),
            encoding="utf-8",
        )
        self.candidate_equations.to_csv(self._get_candidates_path(path))
        self.fidelity_frame.to_csv(self._get_fidelity_path(path))

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        symbol_map = {
            feature_name: sympy.Symbol(feature_name)
            for feature_name in self.feature_names
        }
        expression = sympy.sympify(self.sympy_expression, locals=symbol_map)
        evaluator = sympy.lambdify(
            [symbol_map[feature_name] for feature_name in self.feature_names],
            expression,
            modules="numpy",
        )
        values = evaluator(
            *[
                frame[feature_name].to_numpy(dtype="float64")
                for feature_name in self.feature_names
            ]
        )
        array = np.asarray(values, dtype="float64")
        if array.ndim == 0:
            array = np.full(len(frame), float(array), dtype="float64")
        return pd.Series(
            array.reshape(-1), index=frame.index, name="symbolic_prediction"
        )
