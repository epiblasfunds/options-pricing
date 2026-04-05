# Volatility Model Explainability

This package provides a Dash dashboard for explainability of IBEX option implied-volatility models stored under `src/volatility_models/saved_models/`.

## What It Does

- Discovers explainable-model bundles exported with `src.python_models.explainable_model` and `src.python_models.dashboard_models`.
- Loads the project volatility dataset from `data/volatility_data/VOLATILITY_DB.csv`.
- Derives explainability features such as `Moneyness`, `LogMoneyness`, `ExecHour`, and `ExecWeekday`.
- Exposes four dashboard tabs:
  - Behaviour and surface
  - Global explainability
  - Sample explainability
  - Diagnosis

## Package Structure

- `config.py`: central configuration, default feature schema, default metrics registry, runtime settings.
- `services/shared/`: shared runtime services such as discovery, loading, prediction, dataset access, feature schema, metrics, and cache.
- `services/equivalent_models/`: services for the equivalent-models dashboard tab.
- `services/global_explainability/`: services for the global explainability dashboard tab.
- `services/behaviour_surface/`: services for the behaviour and surface dashboard tab.
- `services/sample_explainability/`: services for the sample explainability dashboard tab.
- `services/diagnosis/`: services for the diagnosis dashboard tab.
- `plots/`: Plotly and tree rendering helpers.
- `dashboard/`: layout, ids, styles, and callbacks.
- `main.py`: application entry point.

## Configuration-Driven Design

The main configuration points are in `config.py`.

- `MODEL_INPUT_FEATURES`
- `CATEGORICAL_FEATURES`
- `NUMERICAL_FEATURES`
- `OPTIONAL_DERIVED_EXPLAINABILITY_FEATURES`
- `TARGET_COLUMN`
- `ERROR_METRICS`

The rest of the package reads from the schema and metric registry instead of scattering feature names and metric names across callbacks.

### Add A New Metric

1. Register it in `_build_default_metrics_registry()` in `config.py`.
2. Add its key to `ERROR_METRICS`.

Example:

```python
registry.register(
    MetricDefinition(
        name="mape",
        label="MAPE",
        function=my_mape_function,
        higher_is_better=False,
        formatter=lambda value: f"{value:.2%}",
    )
)
ERROR_METRICS = ["rmse", "mae", "r2", "mape"]
```

### Add Or Remove An Input Feature

1. Update `MODEL_INPUT_FEATURES`.
2. Update `CATEGORICAL_FEATURES` and `NUMERICAL_FEATURES`.
3. Add or edit the corresponding `FeatureDefinition` in `_build_default_feature_schema()`.

### Add A Derived Explainability Feature

1. Add its name to `OPTIONAL_DERIVED_EXPLAINABILITY_FEATURES`.
2. Add a `FeatureDefinition` for it in `_build_default_feature_schema()`.
3. Extend `utils/feature_utils.py` so the feature is derived from the real dataset without altering the raw model inputs.

## Model Loading

The dashboard only loads explainable-model bundle directories that include precomputed dashboard artifacts:

- `metadata.json`
- `dashboard_model/metadata.json`
- `dashboard_model/dataset_frame.joblib`
- `dashboard_model/global_shap.joblib`
- `dashboard_model/local_shap.joblib`
- `dashboard_model/neighbors_frame.joblib`
- `dashboard_model/surfaces_frame.joblib`
- `dashboard_model/ice_frame.joblib`
- `dashboard_model/ale_frame.joblib`
- `dashboard_model/diagnosis.joblib`
- `dashboard_model/manual_api_stub.joblib`
- `dashboard_model/tree_models/depth_*/...`

Optional bundle sidecar artifacts:

- `preprocessor.joblib`
- `history.json`
- `tuning.json`
- `validation_predictions.csv`
- `test_predictions.csv`

The dashboard does not execute Keras models at runtime. If a bundle does not contain `dashboard_model/`, it is ignored by discovery and must be regenerated.

## Data Access

The package prefers the real project dataset at:

- `data/volatility_data/VOLATILITY_DB.csv`

If that file does not exist, it falls back to the repository loader.

## Running The App

```bash
python -m src.volatility_models.model_explainability.main
```

The app starts on `http://127.0.0.1:8050`.

## Limitations

- The financial checks are heuristic continuity and spike checks, not full no-arbitrage validation.
- Manual input currently calls a stubbed API contract and displays a placeholder response.

## Recommended Bundle Metadata

For smooth deployment, persist bundle metadata with fields like:

```json
{
  "model_input_features": [
    "TimeToExpiration",
    "Rate",
    "UnderlyingPrice",
    "StrikePrice",
    "OptionType",
    "ExecHour",
    "ExecWeekday"
  ],
  "error_metrics": ["rmse", "mae", "r2"],
  "available_surrogate_depths": [2, 4, 8, 16],
  "artifact_files": {
    "dashboard_model": "dashboard_model"
  }
}
```
