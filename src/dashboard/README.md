This package provides the Dash dashboard for volatility-model explainability.

Key directories:
- `src/dashboard/saved_models`: persisted explainable bundles consumed by the UI.
- `src/dashboard/dashboard`: layout and callbacks.
- `src/dashboard/services`: bundle-backed services used by the UI.

The dashboard no longer uses a dedicated runtime module. Configuration is loaded from
`src.config.config.config.dashboard_models_config`, while model features are generated
from raw trades through `src.volatility_models.feature_engineering`.

Launch with:

```bash
python -m src.dashboard.main
```
