from enum import StrEnum


class ModelFormatEnum(StrEnum):
    """The format of the model file. This is used to determine how to load the model file."""

    EXPLAINABLE_MODEL = "explainable_model"
    JOBLIB = "joblib"
    KERAS = "keras"
    H5 = "h5"
