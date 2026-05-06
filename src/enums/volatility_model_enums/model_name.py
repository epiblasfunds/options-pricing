from enum import StrEnum


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
