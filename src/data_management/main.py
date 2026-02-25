from src.data_management.loaders import (
    MergeRawStepLoader,
    ProductSplitStepLoader,
    ReadRawStepLoader,
    UnderlyingStepLoader,
)


def run_data_pipeline():
    for loader in [
        ReadRawStepLoader,
        MergeRawStepLoader,
        ProductSplitStepLoader,
        UnderlyingStepLoader,
    ]:
        loader.load()


if __name__ == "__main__":
    run_data_pipeline()
