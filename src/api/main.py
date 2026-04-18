from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException

from src.api.models import ModelRequest
from src.api.models import PredictionResponse
from src.api.models import SampleExplainabilityResponse
from src.api.services import ApiModelCache
from src.api.services import ApiModelService
from src.api.services import ModelStorage
from src.config.config import config

app = FastAPI(title="Volatility Model API")

_service = ApiModelService(
    storage=ModelStorage(config.clientserver_config),
    cache=ApiModelCache(max_entries=config.clientserver_config.api_cache_entries),
    neighbors_k=config.clientserver_config.dashboard_sample_explainability_neighbors_k,
)


def get_model_service() -> ApiModelService:
    return _service


@app.post("/run_model/predict/", response_model=PredictionResponse)
def predict(
    request: ModelRequest,
    service: ApiModelService = Depends(get_model_service),
):
    try:
        return ApiModelService._json_safe(service.predict(request))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/run_model/sample_explainability/",
    response_model=SampleExplainabilityResponse,
)
def sample_explainability(
    request: ModelRequest,
    service: ApiModelService = Depends(get_model_service),
):
    try:
        return ApiModelService._json_safe(service.sample_explainability(request))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
