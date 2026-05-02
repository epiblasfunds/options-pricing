from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from src.enums.volatility_model_enums import ModelNameEnum


class ApiOptionTypeEnum(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


class PredictionFeatures(BaseModel):
    optionContractCode: str | None = None
    optionType: ApiOptionTypeEnum
    strikePrice: float
    underlyingPrice: float
    timeToExpiration: float
    rate: float
    impliedVolatility: float | None = None


class ModelRequest(BaseModel):
    modelo: ModelNameEnum
    caracteristicas: PredictionFeatures


class PredictionResponse(BaseModel):
    modelo: ModelNameEnum
    prediction: float
    input: dict[str, Any]


class SampleExplainabilityResponse(BaseModel):
    modelo: ModelNameEnum
    prediction: float
    input: dict[str, Any]
    reference_sample_index: Any | None
    waterfall_image: str | None
    local_explanation: dict[str, Any]
    neighbors: list[dict[str, Any]]
    neighbor_distances: list[dict[str, Any]]
