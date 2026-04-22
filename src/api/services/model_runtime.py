from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from enum import Enum
import math
from typing import Any

import numpy as np
import pandas as pd

from src.api.models import ApiOptionTypeEnum
from src.api.models import ModelRequest
from src.api.services.cache import ApiModelCache
from src.api.services.storage import ModelStorage
from src.dashboard.domain import build_feature_schema
from src.dashboard.plots.shap_plots import waterfall_image
from src.dashboard.services.global_explainability import ShapExplanationResult
from src.dashboard.services.shared.feature_schema import FeatureSchema
from src.model2dashboard.features import add_dashboard_derived_features
from src.model2dashboard.features import build_feature_frame_from_trades
from src.model2dashboard.model_io import load_training_runtime
from src.model2dashboard.model_io import predict_raw_frame
from src.python_models.dashboard.artifacts import StoredShapExplanation
from src.python_models.dashboard.dashboard_model import DashboardModel


@dataclass(frozen=True)
class LoadedApiModel:
    training_runtime: Any
    dashboard_model: DashboardModel


class ApiModelService:
    def __init__(
        self,
        *,
        storage: ModelStorage,
        cache: ApiModelCache[LoadedApiModel],
        feature_schema: FeatureSchema | None = None,
        neighbors_k: int = 10,
    ) -> None:
        self.storage = storage
        self.cache = cache
        self.feature_schema = feature_schema or build_feature_schema()
        self.neighbors_k = int(neighbors_k)

    def predict(self, request: ModelRequest) -> dict[str, Any]:
        loaded = self._load_model(request.modelo.value)
        raw_frame = self._request_to_raw_frame(request)
        prediction = float(predict_raw_frame(loaded.training_runtime, raw_frame)[0])
        return {
            "modelo": request.modelo.value,
            "prediction": prediction,
            "input": self._json_safe(raw_frame.iloc[0].to_dict()),
        }

    def sample_explainability(self, request: ModelRequest) -> dict[str, Any]:
        loaded = self._load_model(request.modelo.value)
        raw_frame = self._request_to_raw_frame(request)
        prediction = float(predict_raw_frame(loaded.training_runtime, raw_frame)[0])
        sample_frame = self._build_dashboard_sample_frame(raw_frame, prediction)
        dashboard_model = loaded.dashboard_model

        neighbors = self._find_runtime_neighbors(
            dashboard_model=dashboard_model,
            sample_frame=sample_frame,
            k=self.neighbors_k,
        )
        reference_index = self._nearest_explainable_index(
            dashboard_model=dashboard_model,
            sample_frame=sample_frame,
        )
        explanation_payload: dict[str, Any] = {}
        waterfall_src = None
        if reference_index is not None and dashboard_model.local_shap is not None:
            stored = dashboard_model.local_shap_for_index(reference_index)
            explanation_result = self._stored_shap_to_result(stored)
            waterfall_src = waterfall_image(
                explanation_result,
                reference_index,
                self.feature_schema,
            )
            explanation_payload = self._stored_shap_to_payload(stored)

        neighbor_distances = [
            {"row_id": str(index), "distance": row["distance"]}
            for index, row in neighbors.iterrows()
        ]
        return {
            "modelo": request.modelo.value,
            "prediction": prediction,
            "input": self._json_safe(raw_frame.iloc[0].to_dict()),
            "reference_sample_index": self._json_safe(reference_index),
            "waterfall_image": waterfall_src,
            "local_explanation": self._json_safe(explanation_payload),
            "neighbors": self._frame_records(neighbors),
            "neighbor_distances": self._json_safe(neighbor_distances),
        }

    def _load_model(self, model_name: str) -> LoadedApiModel:
        return self.cache.get_or_load(model_name, lambda: self._load_uncached(model_name))

    def _load_uncached(self, model_name: str) -> LoadedApiModel:
        prepared = self.storage.prepare_model(model_name)
        training_runtime = load_training_runtime(
            family_name=model_name,
            trained_models_dir=prepared.trained_models_dir,
            retrained_metadata_dir=prepared.retrained_metadata_dir,
        )
        dashboard_model = DashboardModel.load(prepared.dashboard_model_dir)
        return LoadedApiModel(
            training_runtime=training_runtime,
            dashboard_model=dashboard_model,
        )

    def _request_to_raw_frame(self, request: ModelRequest) -> pd.DataFrame:
        features = request.caracteristicas
        option_type = "C" if features.optionType == ApiOptionTypeEnum.CALL else "P"
        return pd.DataFrame(
            [
                {
                    "ExecDatetime": self._format_exec_datetime(
                        features.execDatetime
                    ),
                    "OptionType": option_type,
                    "Quantity": int(features.quantity),
                    "StrikePrice": float(features.strikePrice),
                    "TradeType": str(features.tradeType),
                    "UnderlyingLagMinutes": float(features.underlyingLag),
                    "UnderlyingPrice": float(features.underlyingPrice),
                    "TimeToExpiration": float(features.timeToExpiration),
                    "Rate": float(features.rate),
                }
            ]
        )

    def _build_dashboard_sample_frame(
        self,
        raw_frame: pd.DataFrame,
        prediction: float,
    ) -> pd.DataFrame:
        sample = raw_frame.copy()
        feature_frame = build_feature_frame_from_trades(sample)
        for column in feature_frame.columns:
            sample[column] = feature_frame[column].to_numpy()
        sample = add_dashboard_derived_features(sample)
        sample["PredictedVolatility"] = prediction
        return sample

    def _find_runtime_neighbors(
        self,
        *,
        dashboard_model: DashboardModel,
        sample_frame: pd.DataFrame,
        k: int,
    ) -> pd.DataFrame:
        dataset = dashboard_model.dataset_frame
        feature_names = self._neighbor_feature_names(dashboard_model, sample_frame)
        if not feature_names:
            return pd.DataFrame()
        dataset_matrix = dataset.loc[:, feature_names].apply(
            pd.to_numeric,
            errors="coerce",
        )
        sample_vector = sample_frame.loc[:, feature_names].iloc[0].apply(
            pd.to_numeric,
            errors="coerce",
        )
        center = dataset_matrix.mean()
        scale = dataset_matrix.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        normalized_dataset = (dataset_matrix.fillna(center) - center) / scale
        normalized_sample = (sample_vector.fillna(center) - center) / scale
        distances = np.sqrt(
            ((normalized_dataset - normalized_sample.to_numpy()) ** 2).mean(axis=1)
        )
        neighbor_indices = distances.sort_values().head(k).index
        neighbors = dataset.loc[neighbor_indices].copy()
        neighbors["distance"] = distances.loc[neighbor_indices].to_numpy()
        return neighbors

    def _nearest_explainable_index(
        self,
        *,
        dashboard_model: DashboardModel,
        sample_frame: pd.DataFrame,
    ) -> Any | None:
        if dashboard_model.local_shap is None or not dashboard_model.local_shap.index:
            return None
        dataset = dashboard_model.dataset_frame
        explainable_indices = [
            index for index in dashboard_model.local_shap.index if index in dataset.index
        ]
        if not explainable_indices:
            return None
        feature_names = self._neighbor_feature_names(dashboard_model, sample_frame)
        if not feature_names:
            return None
        explainable_frame = dataset.loc[explainable_indices, feature_names]
        sample_vector = sample_frame.loc[:, feature_names].iloc[0]
        center = explainable_frame.mean()
        scale = explainable_frame.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        normalized_frame = (explainable_frame.fillna(center) - center) / scale
        normalized_sample = (sample_vector.fillna(center) - center) / scale
        distances = np.sqrt(
            ((normalized_frame - normalized_sample.to_numpy()) ** 2).mean(axis=1)
        )
        return distances.sort_values().index[0]

    def _neighbor_feature_names(
        self,
        dashboard_model: DashboardModel,
        sample_frame: pd.DataFrame,
    ) -> list[str]:
        candidates = (
            dashboard_model.transformed_feature_names
            or dashboard_model.metadata.get("model_input_features", [])
        )
        return [
            name
            for name in candidates
            if name in dashboard_model.dataset_frame.columns and name in sample_frame.columns
        ]

    @staticmethod
    def _stored_shap_to_result(stored: StoredShapExplanation) -> ShapExplanationResult:
        explain_frame = pd.DataFrame(
            stored.data,
            index=stored.index,
            columns=stored.feature_names,
        )
        return ShapExplanationResult(
            method=stored.method,
            explanation=stored.to_explanation(),
            explain_frame=explain_frame,
            predictions=pd.Series(
                stored.predictions,
                index=stored.index,
                name="PredictedVolatility",
            ),
            mean_abs_shap=pd.Series(stored.mean_abs_shap).sort_values(ascending=False),
            feature_names=list(stored.feature_names),
        )

    @staticmethod
    def _stored_shap_to_payload(stored: StoredShapExplanation) -> dict[str, Any]:
        return {
            "method": stored.method,
            "feature_names": list(stored.feature_names),
            "index": list(stored.index),
            "values": np.asarray(stored.values).tolist(),
            "base_values": np.asarray(stored.base_values).tolist(),
            "data": np.asarray(stored.data).tolist(),
            "predictions": np.asarray(stored.predictions).tolist(),
            "mean_abs_shap": dict(stored.mean_abs_shap),
        }

    @staticmethod
    def _format_exec_datetime(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    @classmethod
    def _frame_records(cls, frame: pd.DataFrame) -> list[dict[str, Any]]:
        return cls._json_safe(frame.reset_index(names="index").to_dict("records"))

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Enum):
            return cls._json_safe(value.value)
        if isinstance(value, np.generic):
            return cls._json_safe(value.item())
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return cls._json_safe(value.tolist())
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            if value.tzinfo is None:
                return value.isoformat()
            return value.tz_convert(timezone.utc).isoformat()
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value
