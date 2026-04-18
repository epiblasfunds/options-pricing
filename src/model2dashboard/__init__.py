def build_all_explainable_models(*args, **kwargs):
    from src.model2dashboard.pipeline import build_all_explainable_models as _impl

    return _impl(*args, **kwargs)


def build_explainable_model(*args, **kwargs):
    from src.model2dashboard.pipeline import build_explainable_model as _impl

    return _impl(*args, **kwargs)


def run_pipeline(*args, **kwargs):
    from src.model2dashboard.pipeline import run_pipeline as _impl

    return _impl(*args, **kwargs)


def __getattr__(name: str):
    if name == "ExportedDashboardBundle":
        from src.model2dashboard.pipeline import ExportedDashboardBundle

        return ExportedDashboardBundle
    raise AttributeError(name)


__all__ = [
    "ExportedDashboardBundle",
    "build_all_explainable_models",
    "build_explainable_model",
    "run_pipeline",
]
