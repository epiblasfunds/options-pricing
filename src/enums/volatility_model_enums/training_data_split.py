from enum import StrEnum


class TrainingDataSplitEnum(StrEnum):
    TRAIN = "train"
    TRAIN_VAL = "trainval"
    VAL = "val"
    TEST = "test"
