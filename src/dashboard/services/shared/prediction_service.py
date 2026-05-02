"""Access helpers for dashboard-ready volatility bundles."""

import json
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

import pandas as pd

from src.config.config import config
from src.dashboard.services.shared.feature_schema import FeatureSchema
from src.dashboard.services.shared.model_loader import LoadedModelBundle, ModelLoader
from src.dashboard.services.shared.model_registry import ModelRegistry


class PredictionPipelineError(RuntimeError):
    """Raised when the dashboard requests data outside the precalculated bundle."""


class PredictionService:
    """Expose precomputed dashboard artifacts for the selected model."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_loader: ModelLoader,
        feature_schema: FeatureSchema,
    ) -> None:
        self.model_registry = model_registry
        self.model_loader = model_loader
        self.feature_schema = feature_schema

    def load_bundle(self, model_id: str) -> LoadedModelBundle:
        discovered = self.model_registry.get_model(model_id)
        if discovered is None:
            raise FileNotFoundError(f"Model '{model_id}' was not found.")
        return self.model_loader.load(discovered)

    def load_dashboard_model(self, model_id: str):
        bundle = self.load_bundle(model_id)
        return bundle.dashboard_model

    def predict_frame(self, model_id: str, frame: pd.DataFrame) -> pd.Series:
        bundle = self.load_bundle(model_id)
        if self._can_use_precomputed_predictions(bundle.dashboard_model, frame):
            return bundle.dashboard_model.predictions_for_indices(frame.index)
        raise PredictionPipelineError(
            "Runtime prediction is disabled in the dashboard. "
            "Only samples already exported inside dashboard_model can be scored here."
        )

    def call_manual_prediction_api(
        self,
        model_id: str,
        sample_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._post_manual_api(
            endpoint="/run_model/predict/",
            model_id=model_id,
            sample_payload=sample_payload,
        )

    def call_manual_sample_explainability_api(
        self,
        model_id: str,
        sample_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._post_manual_api(
            endpoint="/run_model/sample_explainability/",
            model_id=model_id,
            sample_payload=sample_payload,
        )

    @staticmethod
    def _can_use_precomputed_predictions(dashboard_model, frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        return pd.Index(frame.index).isin(dashboard_model.dataset_frame.index).all()

    def _post_manual_api(
        self,
        *,
        endpoint: str,
        model_id: str,
        sample_payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{config.clientserver_config.api_base_url}{endpoint}"
        body = {
            "modelo": model_id,
            "caracteristicas": self._api_features_from_dashboard_sample(sample_payload),
        }
        request = Request(
            url=url,
            data=json.dumps(body, default=str).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=config.clientserver_config.api_timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(
                f"Manual API call failed ({exc.code}): {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Manual API is not reachable at {url}: {exc}") from exc

    def _api_features_from_dashboard_sample(
        self,
        sample_payload: dict[str, Any],
    ) -> dict[str, Any]:
        option_type = sample_payload.get("OptionType")
        if hasattr(option_type, "value"):
            option_type = option_type.value
        option_type_text = str(option_type).upper()
        if option_type_text == "C":
            option_type_text = "CALL"
        if option_type_text == "P":
            option_type_text = "PUT"

        mapping = {
            "ExecDatetime": "execDatetime",
            "OptionContractCode": "optionContractCode",
            "OptionType": "optionType",
            "StrikePrice": "strikePrice",
            "UnderlyingPrice": "underlyingPrice",
            "TimeToExpiration": "timeToExpiration",
            "Rate": "rate",
            "ImpliedVolatility": "impliedVolatility",
        }
        result: dict[str, Any] = {}
        for dashboard_name, api_name in mapping.items():
            if dashboard_name not in sample_payload:
                continue
            value = sample_payload[dashboard_name]
            if value in (None, ""):
                continue
            if dashboard_name == "OptionType":
                value = option_type_text
            result[api_name] = value
        return result
