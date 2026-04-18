import pandas as pd
import sympy
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

from src.config.config import config
from src.dashboard.domain import build_metrics_registry
from src.dashboard.utils.sampling import sample_frame
from src.functionalities.dashboard_models.runtime import predict_raw_frame
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel
from src.volatility_models import build_feature_frame_from_trades

METRICS_REGISTRY = build_metrics_registry()


def build_symbolic_regressor_model(
    *,
    trained_model,
    dataset_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    model_input_features: list[str],
) -> SymbolicRegressorModel:
    sampled = sample_frame(
        dataset_frame,
        max_rows=config.dashboard_models_config.symbolic_sample_size,
        random_state=config.dashboard_models_config.random_state + 5,
    )
    sampled_raw = raw_frame.loc[sampled.index].copy()
    sampled_features = build_feature_frame_from_trades(sampled_raw).loc[
        :, model_input_features
    ]
    sampled_predictions = pd.Series(
        predict_raw_frame(trained_model, sampled_raw),
        index=sampled.index,
        name="model_prediction",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        sampled_features,
        sampled_predictions,
        test_size=0.2,
        random_state=config.dashboard_models_config.random_state,
    )
    regressor = PySRRegressor(
        model_selection="accuracy",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cube"],
        niterations=config.dashboard_models_config.symbolic_niterations,
        populations=config.dashboard_models_config.symbolic_populations,
        population_size=config.dashboard_models_config.symbolic_population_size,
        maxsize=config.dashboard_models_config.symbolic_maxsize,
        maxdepth=config.dashboard_models_config.symbolic_maxdepth,
        timeout_in_seconds=config.dashboard_models_config.symbolic_timeout_seconds,
        random_state=config.dashboard_models_config.random_state,
        deterministic=False,
        batching=True,
        batch_size=min(256, len(X_train)),
        precision=32,
        progress=False,
        verbosity=0,
        update=False,
        parallelism="multithreading",
    )
    regressor.fit(
        X_train.to_numpy(dtype="float32"),
        y_train.to_numpy(dtype="float32"),
        variable_names=model_input_features,
    )
    symbolic_predictions = pd.Series(
        regressor.predict(X_test.to_numpy(dtype="float32")),
        index=y_test.index,
        name="symbolic_prediction",
    )
    metrics = METRICS_REGISTRY.compute_metrics(
        y_test.reset_index(drop=True),
        symbolic_predictions.reset_index(drop=True),
        config.dashboard_models_config.error_metrics,
    )
    candidate_equations = _normalize_equation_table(regressor)
    best_equation = regressor.get_best()
    sympy_expression = regressor.sympy()
    expression = sympy.sympify(str(sympy_expression))
    used_feature_names = sorted(
        feature_name
        for feature_name in model_input_features
        if sympy.Symbol(feature_name) in expression.free_symbols
    )
    return SymbolicRegressorModel(
        equation=str(best_equation["equation"]),
        sympy_expression=str(sympy_expression),
        latex_expression=str(regressor.latex(precision=4)),
        interpretation=_build_interpretation(
            used_feature_names=used_feature_names,
            metrics=metrics,
            best_equation=best_equation,
        ),
        feature_names=list(model_input_features),
        used_feature_names=used_feature_names,
        complexity=int(best_equation["complexity"]),
        model_selection=str(regressor.model_selection),
        metrics={name: float(value) for name, value in metrics.items()},
        candidate_equations=candidate_equations,
        fidelity_frame=pd.DataFrame(
            {
                "model_prediction": y_test,
                "symbolic_prediction": symbolic_predictions,
                "residual": y_test - symbolic_predictions,
            }
        ).reset_index(drop=True),
    )


def _normalize_equation_table(regressor: PySRRegressor) -> pd.DataFrame:
    equations = regressor.equations_
    frame = equations[0].copy() if isinstance(equations, list) else equations.copy()
    keep_columns = [
        name
        for name in ("complexity", "loss", "score", "equation")
        if name in frame.columns
    ]
    normalized = frame.loc[:, keep_columns].copy().reset_index(drop=True)
    normalized["complexity"] = normalized["complexity"].astype("int64")
    normalized["loss"] = normalized["loss"].astype("float64")
    if "score" in normalized.columns:
        normalized["score"] = normalized["score"].astype("float64")
    normalized["selected"] = False
    normalized.loc[int(regressor.get_best().name), "selected"] = True
    normalized["equation"] = normalized["equation"].astype(str)
    return normalized


def _build_interpretation(
    *,
    used_feature_names: list[str],
    metrics: dict[str, float],
    best_equation: pd.Series,
) -> str:
    dominant_terms = ", ".join(used_feature_names[:5]) or "an intercept-like constant"
    return (
        "The symbolic surrogate approximates the trained model with "
        f"RMSE {metrics['rmse']:.4f}. "
        f"It uses {dominant_terms}, reaches complexity "
        f"{int(best_equation['complexity'])}."
    )
