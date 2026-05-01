import base64
import io
from fractions import Fraction

import plotly.express as px
import plotly.graph_objects as go
import sympy
from matplotlib import pyplot as plt
from sympy.printing.latex import LatexPrinter
from sympy.printing.str import StrPrinter
from sympy.parsing.sympy_parser import convert_xor
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations

from src.dashboard.plots.plot_style import HOVERLABEL_STYLE
from src.dashboard.plots.plot_style import STANDARD_MARGIN
from src.dashboard.plots.plot_style import STANDARD_TEMPLATE
from src.dashboard.utils.feature_utils import display_feature_label
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel

plt.switch_backend("Agg")

_SYMBOLIC_FORMULA_NAME_ALIASES = {
    "OptionType": "OptionType",
    "Rate": "Rate",
    "StrikePrice": "Strike",
    "TimeToExpiration": "TimeToExpiration",
    "UnderlyingPrice": "Underlying",
}
_SYMBOLIC_FORMULA_SYMBOL_ORDER = {
    "OptionType": 0,
    "Rate": 1,
    "StrikePrice": 2,
    "TimeToExpiration": 3,
    "UnderlyingPrice": 4,
}
_SYMBOLIC_PARSE_LOCALS = {
    "square": lambda value: value**2,
    "cube": lambda value: value**3,
    "pow": lambda base, exponent: base**exponent,
}
_SYMBOLIC_PARSE_TRANSFORMATIONS = standard_transformations + (convert_xor,)


class _SymbolicTextPrinter(StrPrinter):
    def _print_Float(self, expr):
        return _format_decimal_or_scientific(float(expr))


class _SymbolicLatexPrinter(LatexPrinter):
    def _print_Float(self, expr):
        value = float(expr)
        magnitude = abs(value)
        if magnitude > 0.0 and magnitude < 0.01:
            mantissa, exponent = f"{value:.2e}".split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            return rf"{mantissa} \cdot 10^{{{int(exponent)}}}"
        return _format_decimal_or_scientific(value)


_TEXT_PRINTER = _SymbolicTextPrinter()
_LATEX_PRINTER = _SymbolicLatexPrinter()


def symbolic_frontier_figure(
    model: SymbolicRegressorModel,
    schema=None,
):
    if model.candidate_equations.empty:
        return _empty_figure("No symbolic-equation frontier was persisted.")
    frame = model.candidate_equations.copy()
    if "score" not in frame.columns:
        frame["score"] = 0.0
    frame["display_equation"] = frame["equation"].map(
        lambda value: format_symbolic_equation_text(value, schema=schema)
    )
    fig = px.scatter(
        frame,
        x="complexity",
        y="loss",
        color="selected",
        hover_data={"display_equation": True, "score": True, "selected": False},
        title="Equation Frontier",
        color_discrete_map={True: "#17304f", False: "#7aa5d2"},
    )
    fig.update_traces(
        marker={"size": 12, "line": {"width": 1, "color": "white"}},
        hovertemplate=(
            "Complexity: %{x}<br>"
            "Loss: %{y:.6f}<br>"
            "Score: %{customdata[1]:.4f}<br>"
            "Equation: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=STANDARD_MARGIN,
        xaxis_title="Complexity",
        yaxis_title="Loss",
        hoverlabel=HOVERLABEL_STYLE,
        showlegend=False,
    )
    return fig


def symbolic_fidelity_figure(model: SymbolicRegressorModel):
    if model.fidelity_frame.empty:
        return _empty_figure("No fidelity sample was persisted for the symbolic model.")
    frame = model.fidelity_frame
    diagonal_min = min(
        float(frame["model_prediction"].min()),
        float(frame["symbolic_prediction"].min()),
    )
    diagonal_max = max(
        float(frame["model_prediction"].max()),
        float(frame["symbolic_prediction"].max()),
    )
    fig = px.scatter(
        frame,
        x="model_prediction",
        y="symbolic_prediction",
        title="Symbolic Fidelity",
        opacity=0.62,
    )
    fig.add_trace(
        go.Scatter(
            x=[diagonal_min, diagonal_max],
            y=[diagonal_min, diagonal_max],
            mode="lines",
            line={"color": "#17304f", "dash": "dash"},
            name="Perfect agreement",
            hoverinfo="skip",
        )
    )
    fig.update_traces(
        hovertemplate=(
            "Original model: %{x:.4f}<br>Symbolic surrogate: %{y:.4f}<extra></extra>"
        ),
        selector={"mode": "markers"},
    )
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=STANDARD_MARGIN,
        xaxis_title="Original model prediction",
        yaxis_title="Symbolic surrogate prediction",
        hoverlabel=HOVERLABEL_STYLE,
    )
    return fig


def symbolic_expression_tree_figure(model: SymbolicRegressorModel, schema=None):
    expression = _format_expression_for_display(model.sympy_expression)
    nodes = []
    edges = []
    next_x = 0
    depth_step = 1.8
    leaf_step = 1.5

    def visit(node, depth, parent_id=None):
        nonlocal next_x
        children = list(getattr(node, "args", ()))
        child_ids = [visit(child, depth + 1) for child in children]
        if child_ids:
            x = sum(nodes[child_id]["x"] for child_id in child_ids) / len(child_ids)
        else:
            x = float(next_x)
            next_x += leaf_step
        node_id = len(nodes)
        nodes.append(
            {
                "x": x,
                "y": -float(depth) * depth_step,
                "label": _node_label(node, schema=schema),
                "depth": depth,
            }
        )
        for child_id in child_ids:
            edges.append((node_id, child_id))
        if parent_id is not None:
            edges.append((parent_id, node_id))
        return node_id

    root_id = visit(expression, 0)
    if edges:
        deduped_edges = []
        seen = set()
        for source, target in edges:
            edge = (source, target)
            if edge in seen:
                continue
            seen.add(edge)
            deduped_edges.append(edge)
        edges[:] = deduped_edges
    edge_x = []
    edge_y = []
    for source, target in edges:
        edge_x.extend([nodes[source]["x"], nodes[target]["x"], None])
        edge_y.extend([nodes[source]["y"], nodes[target]["y"], None])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#7aa5d2", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[node["x"] for node in nodes],
            y=[node["y"] for node in nodes],
            mode="markers",
            marker={
                "size": 24,
                "color": "#17304f",
                "line": {"color": "#ffffff", "width": 1},
            },
            customdata=[node["label"] for node in nodes],
            hovertemplate="Operation: %{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    for node in nodes:
        fig.add_annotation(
            x=node["x"],
            y=node["y"] - 0.45,
            text=node["label"],
            textangle=-90,
            showarrow=False,
            font={"size": 11, "color": "#17304f"},
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(23,48,79,0.14)",
            borderpad=2,
        )
    max_depth = max((node["depth"] for node in nodes), default=0)
    fig.update_layout(
        template=STANDARD_TEMPLATE,
        margin=STANDARD_MARGIN,
        title="Expression Operation Tree",
        height=360 + max_depth * 110,
        xaxis={"visible": False},
        yaxis={
            "visible": False,
            "range": [-(max_depth + 1) * depth_step - 1.0, 0.8],
        },
        hoverlabel=HOVERLABEL_STYLE,
    )
    if root_id >= 0:
        fig.update_yaxes(scaleanchor=None)
    return fig


def symbolic_formula_image_src(model: SymbolicRegressorModel, schema=None) -> str:
    latex_expression = format_symbolic_equation_latex(
        model.sympy_expression,
        schema=schema,
    )
    formula_text = f"${latex_expression}$"
    figure = plt.figure(figsize=(13, 1.9), dpi=180)
    figure.patch.set_facecolor("#f7fbff")
    figure.text(
        0.02,
        0.5,
        formula_text,
        fontsize=22,
        va="center",
        ha="left",
        color="#17304f",
    )
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def symbolic_formula_aliases(
    model: SymbolicRegressorModel, schema=None
) -> list[tuple[str, str]]:
    expression = _parse_symbolic_expression(model.sympy_expression)
    aliases = []
    for symbol in _sorted_symbols(expression.free_symbols):
        aliases.append(
            (
                _symbolic_formula_alias(symbol.name),
                display_feature_label(symbol.name, schema)
                if schema is not None
                else symbol.name,
            )
        )
    return aliases


def format_symbolic_equation_text(expression_text, schema=None) -> str:
    try:
        expression = _format_expression_for_display(expression_text)
        return _TEXT_PRINTER.doprint(expression)
    except Exception:
        return str(expression_text)


def format_symbolic_equation_latex(expression_text, schema=None) -> str:
    expression = _format_expression_for_display(expression_text)
    return _LATEX_PRINTER.doprint(expression)


def _format_expression_for_display(expression_text):
    expression = _parse_symbolic_expression(expression_text)
    alias_map = {
        symbol: sympy.Symbol(_symbolic_formula_alias(symbol.name))
        for symbol in expression.free_symbols
    }
    expression = expression.xreplace(alias_map)
    replacements = {
        number: _format_number_atom(number) for number in expression.atoms(sympy.Float)
    }
    return expression.xreplace(replacements)


def _parse_symbolic_expression(expression_text):
    return parse_expr(
        str(expression_text),
        local_dict=_SYMBOLIC_PARSE_LOCALS,
        transformations=_SYMBOLIC_PARSE_TRANSFORMATIONS,
        evaluate=True,
    )


def _format_number_atom(number: sympy.Float):
    value = float(number)
    if abs(value - round(value)) < 1e-10:
        return sympy.Integer(int(round(value)))
    magnitude = abs(value)
    if magnitude > 0.0 and magnitude < 0.01:
        return sympy.Float(f"{value:.2e}")
    rounded_value = round(value, 2)
    rounded_error = abs(rounded_value - value)
    fraction = Fraction(value).limit_denominator(12)
    fraction_value = fraction.numerator / fraction.denominator
    fraction_error = abs(fraction_value - value)
    if (
        fraction.denominator != 1
        and fraction_error + 1e-12 < rounded_error
        and fraction_error <= 0.01
    ):
        return sympy.Rational(fraction.numerator, fraction.denominator)
    return sympy.Float(f"{value:.2f}")


def _format_decimal_or_scientific(value: float) -> str:
    magnitude = abs(value)
    if magnitude > 0.0 and magnitude < 0.01:
        mantissa, exponent = f"{value:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _sorted_symbols(symbols) -> list[sympy.Symbol]:
    return sorted(
        symbols,
        key=lambda item: (
            _SYMBOLIC_FORMULA_SYMBOL_ORDER.get(item.name, 999),
            item.name,
        ),
    )


def _symbolic_formula_alias(feature_name: str) -> str:
    return _SYMBOLIC_FORMULA_NAME_ALIASES.get(feature_name, feature_name)


def _node_label(node, schema=None) -> str:
    if isinstance(node, sympy.Symbol):
        return _symbolic_formula_alias(str(node))
    if node.is_Number:
        formatted_number = (
            node
            if isinstance(node, (sympy.Integer, sympy.Rational))
            else _format_number_atom(sympy.Float(float(node)))
        )
        return _TEXT_PRINTER.doprint(formatted_number)
    label_map = {
        "Add": "+",
        "Mul": "*",
        "Pow": "^",
    }
    return label_map.get(node.func.__name__, node.func.__name__)


def _empty_figure(message: str):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 14, "color": "#4a5a73"},
    )
    fig.update_layout(template=STANDARD_TEMPLATE, margin=STANDARD_MARGIN)
    return fig
