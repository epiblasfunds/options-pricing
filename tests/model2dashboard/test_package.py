import sys
import types

import pytest

import src.model2dashboard as model2dashboard


def test_model2dashboard_package_exposes_lazy_pipeline_wrappers(monkeypatch):
    bundle_type = type("ExportedDashboardBundle", (), {})
    pipeline_stub = types.ModuleType("src.model2dashboard.pipeline")
    pipeline_stub.build_all_explainable_models = lambda *args, **kwargs: (
        "all",
        args,
        kwargs,
    )
    pipeline_stub.build_explainable_model = lambda *args, **kwargs: (
        "one",
        args,
        kwargs,
    )
    pipeline_stub.run_pipeline = lambda *args, **kwargs: ("run", args, kwargs)
    pipeline_stub.ExportedDashboardBundle = bundle_type
    monkeypatch.setitem(sys.modules, "src.model2dashboard.pipeline", pipeline_stub)

    assert model2dashboard.build_all_explainable_models(1, a=2) == (
        "all",
        (1,),
        {"a": 2},
    )
    assert model2dashboard.build_explainable_model(3) == ("one", (3,), {})
    assert model2dashboard.run_pipeline(x=4) == ("run", (), {"x": 4})
    assert model2dashboard.ExportedDashboardBundle is bundle_type
    with pytest.raises(AttributeError):
        getattr(model2dashboard, "missing")
