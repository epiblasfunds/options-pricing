from pathlib import Path
from src.python_models.explainable_artifacts import AbstractModelMetadata
from src.python_models.explainable_artifacts import ExplainableModelMetadata
from src.python_models.explainable_artifacts import SingleModelMetadata
from src.python_models.explainable_artifacts import SurrogateTreeModel
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel
from src.volatility_models.trained_model import TrainedModel


class ExplainableModel:
    def __init__(
        self,
        main_model: TrainedModel,
        tree_models: dict[int, SurrogateTreeModel],
        symbolic_model: SymbolicRegressorModel | None,
        metadata: ExplainableModelMetadata,
    ):
        self.main_model = main_model
        self.tree_models = {int(depth): model for depth, model in tree_models.items()}
        self.symbolic_model = symbolic_model
        self.metadata = metadata

    @classmethod
    def load(cls, metadata: ExplainableModelMetadata) -> "ExplainableModel":
        main_model = TrainedModel.load(metadata=metadata.get_trained_model_metadata())
        tree_models = cls._load_tree_models(metadata)
        symbolic_model = cls._load_symbolic_model(metadata)
        return cls(
            main_model=main_model,
            tree_models=tree_models,
            symbolic_model=symbolic_model,
            metadata=metadata,
        )

    def save(self, path: Path | None = None) -> None:
        bundle_path = path or self.metadata.path
        bundle_path.mkdir(parents=True, exist_ok=True)
        self.metadata.save(bundle_path)
        self.main_model.save(bundle_path / "trained_model")
        tree_models_path = bundle_path / "tree_models"
        tree_models_path.mkdir(parents=True, exist_ok=True)
        for depth, tree_model in sorted(self.tree_models.items()):
            tree_model.save(tree_models_path / f"depth_{int(depth)}")
        if self.symbolic_model is not None:
            self.symbolic_model.save(bundle_path / "symbolic_model")

    @staticmethod
    def _load_tree_models(
        metadata: ExplainableModelMetadata,
    ) -> dict[int, SurrogateTreeModel]:
        tree_models_path = metadata.get_tree_models_path()
        if tree_models_path.exists():
            loaded: dict[int, SurrogateTreeModel] = {}
            for tree_path in sorted(
                tree_models_path.iterdir(), key=lambda path: path.name
            ):
                if not tree_path.is_dir():
                    continue
                try:
                    depth = int(tree_path.name.removeprefix("depth_"))
                except ValueError:
                    continue
                loaded[depth] = SurrogateTreeModel.load(
                    metadata=metadata.get_tree_model_metadata(depth)
                )
            if loaded:
                return loaded

        raise FileNotFoundError(
            f"No persisted surrogate trees were found under '{tree_models_path}'."
        )

    @staticmethod
    def _load_symbolic_model(
        metadata: ExplainableModelMetadata,
    ) -> SymbolicRegressorModel | None:
        symbolic_model_path = metadata.get_symbolic_model_path()
        if symbolic_model_path.exists():
            return SymbolicRegressorModel.load(symbolic_model_path)
        return None


__all__ = [
    "AbstractModelMetadata",
    "ExplainableModel",
    "ExplainableModelMetadata",
    "SingleModelMetadata",
    "SurrogateTreeModel",
]
