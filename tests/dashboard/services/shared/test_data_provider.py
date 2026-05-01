from types import SimpleNamespace

import pandas as pd
import pytest

from src.dashboard.services.shared.data_provider import VolatilityDataProvider


def _dataset_frame():
    return pd.DataFrame(
        {
            "ExecDatetime": ["2026-04-22T10:00:00Z"],
            "OptionType": ["C"],
            "StrikePrice": [9000.0],
            "UnderlyingPrice": [9100.0],
            "TimeToExpiration": [15.0],
            "Rate": [0.02],
            "PredictedVolatility": [0.2],
        },
        index=[5],
    )


def test_data_provider_loads_bundle_dataset_and_uses_cache():
    provider = VolatilityDataProvider()
    expected = _dataset_frame()
    loader_calls = []
    provider.bind_model_runtime(
        model_registry=SimpleNamespace(discover_models=lambda: [object()]),
        model_loader=SimpleNamespace(
            load=lambda discovered: loader_calls.append(discovered) or SimpleNamespace(
                dashboard_model=SimpleNamespace(dataset_frame=expected)
            )
        ),
    )

    first = provider.load_dataset()
    second = provider.load_dataset()

    assert first.equals(expected)
    assert second.equals(first)
    assert len(loader_calls) == 1


def test_data_provider_prefers_requested_bundle_dataset():
    provider = VolatilityDataProvider()
    expected = _dataset_frame()
    provider.bind_model_runtime(
        model_registry=SimpleNamespace(get_model=lambda model_id: object()),
        model_loader=SimpleNamespace(
            load=lambda discovered: SimpleNamespace(
                dashboard_model=SimpleNamespace(dataset_frame=expected)
            )
        ),
    )

    loaded = provider.load_dataset(model_id="random_forest")

    assert loaded.equals(expected)


def test_data_provider_uses_first_discovered_model_when_no_id_is_given():
    provider = VolatilityDataProvider()
    expected = _dataset_frame()
    provider.bind_model_runtime(
        model_registry=SimpleNamespace(discover_models=lambda: [object()]),
        model_loader=SimpleNamespace(
            load=lambda discovered: SimpleNamespace(
                dashboard_model=SimpleNamespace(dataset_frame=expected)
            )
        ),
    )

    loaded = provider.load_dataset()

    assert loaded.equals(expected)


def test_data_provider_raises_when_requested_bundle_does_not_exist():
    provider = VolatilityDataProvider()
    provider.bind_model_runtime(
        model_registry=SimpleNamespace(get_model=lambda model_id: None),
        model_loader=SimpleNamespace(load=lambda discovered: discovered),
    )

    with pytest.raises(FileNotFoundError, match="random_forest"):
        provider.load_dataset(model_id="random_forest")


def test_data_provider_raises_when_no_discovered_bundles_are_available():
    provider = VolatilityDataProvider()
    provider.bind_model_runtime(
        model_registry=SimpleNamespace(discover_models=lambda: []),
        model_loader=SimpleNamespace(load=lambda discovered: discovered),
    )

    with pytest.raises(FileNotFoundError, match="No dashboard model bundles"):
        provider.load_dataset()
