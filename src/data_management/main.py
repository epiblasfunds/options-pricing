from src.data_management.loaders import (
    MergeRawStepLoader,
    ProductSplitStepLoader,
    ReadRatesRawStepLoader,
    ReadRawStepLoader,
    UnderlyingRatesStepLoader,
    UnderlyingStepLoader,
    VolatilityStepLoader,
)


def run_data_pipeline():
    for loader in [
        ReadRawStepLoader,
        MergeRawStepLoader,
        ProductSplitStepLoader,
        UnderlyingStepLoader,
        ReadRatesRawStepLoader,
        UnderlyingRatesStepLoader,
        VolatilityStepLoader
    ]:
        loader.load()


if __name__ == "__main__":
    run_data_pipeline()
