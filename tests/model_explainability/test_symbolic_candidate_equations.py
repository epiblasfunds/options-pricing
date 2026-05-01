import pandas as pd

from src.model2dashboard.artifact_builders import _normalize_symbolic_equation_table


class _FakeRegressor:
    def __init__(self, equations: pd.DataFrame, best_equation: str):
        self.equations_ = equations
        self._best_equation = best_equation

    def get_best(self) -> pd.Series:
        return self.equations_.loc[
            self.equations_["equation"] == self._best_equation
        ].iloc[0]


def test_normalize_symbolic_equation_table_persists_at_least_five_candidates():
    equations = pd.DataFrame(
        [
            {"complexity": 1, "loss": 0.50, "score": 0.00, "equation": "c0"},
            {"complexity": 2, "loss": 0.40, "score": 0.01, "equation": "c1"},
            {"complexity": 2, "loss": 0.39, "score": 0.02, "equation": "c1b"},
            {"complexity": 3, "loss": 0.30, "score": 0.03, "equation": "c2"},
            {"complexity": 4, "loss": 0.25, "score": 0.04, "equation": "c3"},
            {"complexity": 5, "loss": 0.20, "score": 0.05, "equation": "c4"},
            {"complexity": 6, "loss": 0.15, "score": 0.06, "equation": "best_eq"},
            {"complexity": 7, "loss": 0.14, "score": 0.07, "equation": "c6"},
        ]
    )
    regressor = _FakeRegressor(equations=equations, best_equation="best_eq")

    normalized = _normalize_symbolic_equation_table(regressor, min_equations=5)

    assert len(normalized) >= 5
    assert normalized["equation"].is_unique
    assert normalized["selected"].sum() == 1
    assert "best_eq" in set(normalized["equation"])


def test_normalize_symbolic_equation_table_keeps_best_equation_even_if_not_in_first_five():
    equations = pd.DataFrame(
        [
            {"complexity": 1, "loss": 0.60, "score": 0.00, "equation": "eq1"},
            {"complexity": 2, "loss": 0.50, "score": 0.01, "equation": "eq2"},
            {"complexity": 3, "loss": 0.40, "score": 0.02, "equation": "eq3"},
            {"complexity": 4, "loss": 0.30, "score": 0.03, "equation": "eq4"},
            {"complexity": 5, "loss": 0.20, "score": 0.04, "equation": "eq5"},
            {"complexity": 40, "loss": 0.19, "score": 0.90, "equation": "best_eq"},
        ]
    )
    regressor = _FakeRegressor(equations=equations, best_equation="best_eq")

    normalized = _normalize_symbolic_equation_table(regressor, min_equations=5)

    assert "best_eq" in set(normalized["equation"])
    assert normalized.loc[normalized["equation"] == "best_eq", "selected"].item()
