from src.data_management.loaders import (
    MergeRawStepLoader,
    ProductSplitStepLoader,
    ReadRawStepLoader,
    UnderlyingStepLoader,
    VolatilityStepLoader,
)


def run_data_pipeline():
    for loader in [
        ReadRawStepLoader,
        MergeRawStepLoader,
        ProductSplitStepLoader,
        UnderlyingStepLoader,
        VolatilityStepLoader
    ]:
        loader.load()


if __name__ == "__main__":
    run_data_pipeline()
