from enum import StrEnum


class TrainingDataSplitEnum(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"