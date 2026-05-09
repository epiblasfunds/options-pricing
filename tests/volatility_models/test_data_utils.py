import pandas as pd

from src.enums.data_enums.database_schema.volatility_db_enum import VolatilityDBEnum
from src.volatility_models.data_utils import TRAINING_DATA_CONFIG, TrainingDataHandler


def _build_mock_train_df(n_dates: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    for date in dates:
        # Two persistent contracts that span the full timeline.
        rows.append(
            {
                VolatilityDBEnum.EXEC_DATETIME.value: pd.Timestamp(date),
                VolatilityDBEnum.OPTION_CONTRACT_CODE.value: "CROSS_A",
            }
        )
        rows.append(
            {
                VolatilityDBEnum.EXEC_DATETIME.value: pd.Timestamp(date),
                VolatilityDBEnum.OPTION_CONTRACT_CODE.value: "CROSS_B",
            }
        )

    return pd.DataFrame(rows)


def test_load_kfolds_enforces_no_contract_overlap_between_train_and_val(monkeypatch):
    mock_train_df = _build_mock_train_df(n_dates=30)

    monkeypatch.setattr(
        TrainingDataHandler,
        "load_full_features_splitted_data",
        classmethod(lambda cls, verbose=True: (mock_train_df.copy(), pd.DataFrame(), pd.DataFrame())),
    )
    monkeypatch.setattr(TRAINING_DATA_CONFIG.kfolds_config, "n_folds", 3)
    monkeypatch.setattr(TRAINING_DATA_CONFIG.kfolds_config, "extra_blocks", 1)

    n_folds = TRAINING_DATA_CONFIG.kfolds_config.n_folds
    n_blocks = n_folds + TRAINING_DATA_CONFIG.kfolds_config.extra_blocks
    block_size = mock_train_df.shape[0] // n_blocks

    raw_overlap_exists = False
    for i in range(n_folds):
        kfold_full_df = mock_train_df.iloc[: -block_size * i] if i > 0 else mock_train_df
        raw_train_df, raw_val_df = TrainingDataHandler.split_train_test(kfold_full_df)

        raw_train_contracts = set(raw_train_df[VolatilityDBEnum.OPTION_CONTRACT_CODE.value])
        raw_val_contracts = set(raw_val_df[VolatilityDBEnum.OPTION_CONTRACT_CODE.value])
        if raw_train_contracts.intersection(raw_val_contracts):
            raw_overlap_exists = True
            break

    assert raw_overlap_exists

    folds = TrainingDataHandler.load_kfolds(verbose=False)

    for fold_data in folds.values():
        fold_train_df = fold_data["train"]
        fold_val_df = fold_data["val"]

        train_contracts = set(fold_train_df[VolatilityDBEnum.OPTION_CONTRACT_CODE.value])
        val_contracts = set(fold_val_df[VolatilityDBEnum.OPTION_CONTRACT_CODE.value])

        assert train_contracts.isdisjoint(val_contracts)
