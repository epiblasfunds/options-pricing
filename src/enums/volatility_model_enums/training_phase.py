from enum import StrEnum


class TrainingPhase(StrEnum):
    CV = "cv"
    TRAIN_VAL = "train_val"
    FINAL_TEST = "final_test"
