import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.plots.plot_style import HOVERLABEL_STYLE
from src.dashboard.plots.plot_style import STANDARD_MARGIN
from src.dashboard.plots.plot_style import STANDARD_TEMPLATE
from src.python_models.symbolic_regressor_model import SymbolicRegressorModel


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


def _empty_figure(message: str):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 14, "color": "#4a5a73"},
    )
    fig.update_layout(template=STANDARD_TEMPLATE, margin=STANDARD_MARGIN)
    return fig
