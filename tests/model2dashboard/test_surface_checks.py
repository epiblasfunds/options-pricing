import pandas as pd

from src.model2dashboard.surface_checks import financial_checks_from_surface


def test_financial_checks_report_empty_and_large_jumps():
    assert financial_checks_from_surface(pd.DataFrame()) == [
        "No local surface could be generated for the financial checks."
    ]

    surface = pd.DataFrame(
        {
            "TimeToExpiration": [10, 10, 20, 20],
            "Moneyness": [0.9, 1.0, 0.9, 1.0],
            "PredictedVolatility": [0.1, 0.4, 0.45, 0.75],
        }
    )

    warnings = financial_checks_from_surface(surface)

    assert any("smile points" in warning for warning in warnings)
    assert any("maturity points" in warning for warning in warnings)


def test_financial_checks_report_stable_surface_when_no_jumps_exist():
    surface = pd.DataFrame(
        {
            "TimeToExpiration": [10, 10, 20, 20],
            "Moneyness": [0.9, 1.0, 0.9, 1.0],
            "PredictedVolatility": [0.1, 0.11, 0.12, 0.13],
        }
    )

    assert financial_checks_from_surface(surface) == [
        "No large discontinuities were detected by the heuristic checks."
    ]
