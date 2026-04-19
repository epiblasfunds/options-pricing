
#!/bin/bash
set -e

# Forzar gsutil a usar el Python del entorno virtual si está activo
if [ -n "$VIRTUAL_ENV" ]; then
  export CLOUDSDK_PYTHON="$VIRTUAL_ENV/Scripts/python.exe"
fi

# Configura aquí los nombres de tus buckets
VOLATILITY_BUCKET="options-pricing-explainability-volatility"
DASHBOARD_BUCKET="options-pricing-explainability-dashboard"


# Sube los modelos de volatility
if [ -d src/volatility_models/trained_models ]; then
  echo "Subiendo modelos de volatility a GCP..."
  gcloud storage cp --recursive src/volatility_models/trained_models gs://$VOLATILITY_BUCKET/
else
  echo "No existe src/volatility_models/trained_models, omitiendo."
fi

echo "Subida finalizada."
