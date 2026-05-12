from src.dashboard.plots.symbolic_plots import format_symbolic_equation_latex
from src.dashboard.plots.symbolic_plots import format_symbolic_equation_text
from src.dashboard.plots.symbolic_plots import symbolic_formula_aliases
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel


def test_format_symbolic_equation_text_uses_readable_names_and_compact_coefficients():
    expression = (
        "0.333333333333*Rate + 3.30266851204e-5*UnderlyingPrice "
        "- 0.125*StrikePrice + 0.000177841732748*TimeToExpiration "
        "+ 0.0215162951222*OptionType - 0.176559849524"
    )

    formatted = format_symbolic_equation_text(expression)

    assert "x_" not in formatted
    assert "StrikePrice" not in formatted
    assert "UnderlyingPrice" not in formatted
    assert "optionType" in formatted
    assert "strike" in formatted
    assert "underlying" in formatted
    assert "rate/3" in formatted
    assert "strike/8" in formatted
    assert "1.78e-4*timeToExpiration" in formatted
    assert "3.3e-5*underlying" in formatted


def test_format_symbolic_equation_latex_supports_fractions_and_scientific_notation():
    expression = "0.333333333333*Rate + 3.30266851204e-5*UnderlyingPrice"

    formatted = format_symbolic_equation_latex(expression)

    assert r"\frac{rate}{3}" in formatted
    assert r"3.3 \cdot 10^{-5} underlying" in formatted


def test_format_symbolic_equation_latex_uses_times_between_feature_products():
    expression = (
        "Rate*StrikePrice + 2*StrikePrice*TimeToExpiration "
        "+ Rate*UnderlyingPrice/3 + 3.30266851204e-5*StrikePrice"
    )

    formatted = format_symbolic_equation_latex(expression)

    assert r"rate \times strike" in formatted
    assert r"2 strike \times timeToExpiration" in formatted
    assert r"\frac{rate \times underlying}{3}" in formatted
    assert r"3.3 \cdot 10^{-5} strike" in formatted
    assert r"3.3 \cdot 10^{-5} \times strike" not in formatted


def test_symbolic_formula_aliases_use_feature_names_instead_of_positional_aliases():
    model = SymbolicRegressorModel(
        equation="OptionType + Rate + StrikePrice + TimeToExpiration + UnderlyingPrice",
        sympy_expression="OptionType + Rate + StrikePrice + TimeToExpiration + UnderlyingPrice",
        latex_expression="",
        interpretation="",
        feature_names=[],
        used_feature_names=[],
        complexity=5,
        model_selection="best",
    )

    aliases = symbolic_formula_aliases(model)

    assert [alias for alias, _ in aliases] == [
        "optionType",
        "rate",
        "strike",
        "timeToExpiration",
        "underlying",
    ]
