import base64
import io

import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import sympy

from src.dashboard.plots.plot_style import HOVERLABEL_STYLE
from src.dashboard.plots.plot_style import STANDARD_MARGIN
from src.dashboard.plots.plot_style import STANDARD_TEMPLATE
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def symbolic_frontier_figure(model: SymbolicRegressorModel):
    if model.candidate_equations.empty:
        return _empty_figure("No symbolic-equation frontier was persisted.")
    frame = model.candidate_equations.copy()
    fig = px.scatter(
        frame,
        x="complexity",
        y="loss",
        color="selected",
        hover_data={"equation": True, "score": True, "selected": False},
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
            "Original model: %{x:.4f}<br>"
            "Symbolic surrogate: %{y:.4f}<extra></extra>"
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


def symbolic_expression_tree_figure(model: SymbolicRegressorModel):
    expression = sympy.sympify(model.sympy_expression)
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
                "label": _node_label(node),
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


def symbolic_formula_image_src(model: SymbolicRegressorModel) -> str:
    latex_expression, _ = _aliased_formula(model)
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


def symbolic_formula_aliases(model: SymbolicRegressorModel) -> list[tuple[str, str]]:
    _, aliases = _aliased_formula(model)
    return aliases


def _aliased_formula(model: SymbolicRegressorModel) -> tuple[str, list[tuple[str, str]]]:
    expression = sympy.sympify(model.sympy_expression)
    symbols = sorted(expression.free_symbols, key=lambda item: item.name)
    alias_map = {
        symbol: sympy.Symbol(f"x_{index}") for index, symbol in enumerate(symbols, start=1)
    }
    aliased_expression = expression.xreplace(alias_map)
    latex_expression = sympy.latex(
        aliased_expression,
        fold_short_frac=False,
        fold_frac_powers=False,
        long_frac_ratio=2,
    )
    aliases = [
        (f"x_{index}", symbol.name) for index, symbol in enumerate(symbols, start=1)
    ]
    return latex_expression, aliases


def _node_label(node) -> str:
    if isinstance(node, sympy.Symbol):
        return str(node)
    if node.is_Number:
        return str(round(float(node), 4))
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
