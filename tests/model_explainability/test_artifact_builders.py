from types import SimpleNamespace

import numpy as np
import pandas as pd
import shap

from src.model2dashboard import artifact_builders


class _FakeExplainer:
    def __init__(
        self,
        model,
        masker,
        algorithm,
        feature_names,
        seed,
    ) -> None:
        self.model = model
        self.feature_names = feature_names

    def __call__(self, encoded_frame, max_evals, silent):
        predictions = np.asarray(self.model(encoded_frame), dtype="float64").reshape(-1)
        values = np.zeros(
            (len(encoded_frame), len(self.feature_names)), dtype="float64"
        )
        values[:, 0] = predictions
        return shap.Explanation(
            values=values,
            base_values=np.zeros(len(encoded_frame), dtype="float64"),
            data=encoded_frame.to_numpy(),
            feature_names=self.feature_names,
        )


def test_build_shap_artifacts_preserves_row_specific_hidden_inputs(monkeypatch):
    captured_raw_frames = []

    def fake_predict_raw_frame(_runtime, raw_frame):
        captured_raw_frames.append(raw_frame.copy())
        exec_dt = pd.to_datetime(raw_frame["ExecDatetime"], errors="coerce")
        trade_is_h = (raw_frame["TradeType"].astype(str) == "H").astype(float)
        return (
            exec_dt.dt.hour.astype(float) * 100.0
            + raw_frame["UnderlyingLagMinutes"].astype(float) * 10.0
            + raw_frame["Quantity"].astype(float)
            + trade_is_h
        ).to_numpy(dtype="float64")

    monkeypatch.setattr(artifact_builders.shap, "Explainer", _FakeExplainer)
    monkeypatch.setattr(artifact_builders, "predict_raw_frame", fake_predict_raw_frame)
    monkeypatch.setattr(
        artifact_builders,
        "sample_frame",
        lambda frame, max_rows, random_state: frame.head(1),
    )

    raw_frame = pd.DataFrame(
        [
            {
                "ExecDatetime": "2026-04-22T10:00:00+00:00",
                "OptionContractCode": "CIBX 9000X26",
                "OptionType": "C",
                "Quantity": 1,
                "StrikePrice": 9000.0,
                "TradeType": "M",
                "UnderlyingLagMinutes": 0.25,
                "UnderlyingPrice": 9050.0,
                "TimeToExpiration": 15.0,
                "Rate": -0.5,
                "ImpliedVolatility": 0.20,
            },
            {
                "ExecDatetime": "2026-04-23T15:00:00+00:00",
                "OptionContractCode": "PIBX 9100X26",
                "OptionType": "P",
                "Quantity": 7,
                "StrikePrice": 9100.0,
                "TradeType": "H",
                "UnderlyingLagMinutes": 3.5,
                "UnderlyingPrice": 9000.0,
                "TimeToExpiration": 20.0,
                "Rate": -0.6,
                "ImpliedVolatility": 0.21,
            },
        ],
        index=[10, 11],
    )
    dataset_frame = raw_frame.copy()
    predictions = pd.Series(
        [0.2, 0.3], index=raw_frame.index, name="PredictedVolatility"
    )

    _global_shap, local_shap = artifact_builders.build_shap_artifacts(
        runtime=SimpleNamespace(),
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        predictions=predictions,
        sample_indices=[11],
    )

    local_prediction = float(local_shap.waterfall_predictions()[0])

    assert captured_raw_frames, (
        "The SHAP builder should evaluate reconstructed raw rows."
    )
    assert local_prediction == 1543.0
    assert local_shap.index == [11]
