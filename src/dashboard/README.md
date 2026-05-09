This package provides the Dash dashboard for volatility-model explainability.

Key directories:
- `src/volatility_models/trained_models`: trained model artifacts.
- `src/volatility_models/trained_metadata/retrained_metadata`: train-val and final-test metadata.
- `src/dashboard/saved_models`: dashboard-ready explainable bundles consumed by the UI.

Build or rebuild all dashboard bundles with:

```bash
python -m src.model2dashboard.pipeline
```

or from Python:

```python
from src.model2dashboard import run_pipeline

run_pipeline()
```

The bundle builder loads dashboard samples and diagnosis data from
`TrainingDataHandler.load_splitted_data()[-1]`, so those views are generated on the
Test split. Nearest-neighbour search uses
`TrainingDataHandler.load_splitted_data()[0]`, so neighbour tables and maps are
anchored on the Train split. Each generated bundle contains:
- final-test predictions and diagnosis artifacts;
- global and local SHAP values;
- behaviour surfaces, ICE and ALE frames;
- nearest-neighbour frames;
- decision-tree surrogate models;
- a symbolic regression surrogate.

Launch with:

```bash
python -m src.dashboard.main
```
