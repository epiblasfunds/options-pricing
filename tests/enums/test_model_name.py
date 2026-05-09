from src.enums.volatility_model_enums.model_name import display_model_name


def test_display_model_name_removes_explainable_model_suffix():
    assert display_model_name("Random Forest (Explainable Model)") == "Random Forest"


def test_display_model_name_relabels_progressive_as_atm():
    assert (
        display_model_name("xgboost_retrained_progressive_explainable_model")
        == "Xgboost Retrained ATM"
    )
