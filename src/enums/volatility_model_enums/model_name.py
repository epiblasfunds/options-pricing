from enum import StrEnum
import re


class ModelNameEnum(StrEnum):
    LINEAR_REGRESSION = "linear_regression"
    LINEAR_REGRESSION_RETRAINED_PROGRESSIVE = "linear_regression_retrained_progressive"
    QUANTUM_INSPIRED_NN = "quantum_inspired_nn"
    QUANTUM_INSPIRED_NN_RETRAINED_PROGRESSIVE = (
        "quantum_inspired_nn_retrained_progressive"
    )
    RANDOM_FOREST = "random_forest"
    RANDOM_FOREST_RETRAINED_PROGRESSIVE = "random_forest_retrained_progressive"
    SEQUENTIAL_NN = "sequential_nn"
    SEQUENTIAL_NN_RETRAINED_PROGRESSIVE = "sequential_nn_retrained_progressive"
    XGBOOST = "xgboost"
    XGBOOST_RETRAINED_PROGRESSIVE = "xgboost_retrained_progressive"


def display_model_name(model_name_or_id: str) -> str:
    text = str(model_name_or_id).replace("_", " ").strip()
    if not text:
        return text
    text = text.title()
    text = re.sub(r"\bProgressive\b", "ATM", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s*\(?\bExplainable Model\b\)?\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" -_()")
    return text
