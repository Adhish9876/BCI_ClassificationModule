"""
Components — varied, visually rich, high contrast.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from classification.schemas import EvaluationResult, ExperimentResult

C_BLUE    = "#2563EB"
C_NAVY    = "#0F172A"
C_VIOLET  = "#7C3AED"
C_AMBER   = "#D97706"
C_SKY     = "#0284C7"
C_EMERALD = "#059669"
C_SLATE   = "#334155"
C_GRID    = "#E2E8F0"


# ── Header ──────────────────────────────────────────────────────────────────
def render_header(dataset_name: str, source_type: str, is_synthetic: bool = False):
    pill_class = "pill-blue" if source_type in ("sample", "integrated") else "pill-amber"
    pill_text  = ("SAMPLE · SYNTHETIC" if is_synthetic else source_type.upper())
    st.markdown(
        f'<div class="lab-hero">'
        f'<div>'
        f'<div class="lab-hero-title">'
        f'<span>🧠</span>'
        f'<span>BCI Classification Lab</span>'
        f'</div>'
        f'<div class="lab-hero-sub">{dataset_name}</div>'
        f'</div>'
        f'<span class="pill {pill_class}">{pill_text}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


# ── Stepper ──────────────────────────────────────────────────────────────────
def render_stepper(current_step: int):
    steps = [
        ("1", "Load Dataset",    "Pick or upload EEG features"),
        ("2", "Select & Train",  "Configure and run classifiers"),
        ("3", "Results",         "Compare and export results"),
    ]

    def node_class(i):
        if i + 1 < current_step:
            return "done"
        if i + 1 == current_step:
            return "done" if current_step >= 3 else "active"
        return "idle"

    def check(i):
        if i + 1 < current_step:
            return "&#10003;"
        if i + 1 == current_step and current_step >= 3:
            return "&#10003;"
        return steps[i][0]

    html = '<div class="stepper-wrap">'
    for i, (num, label, sub) in enumerate(steps):
        cls = node_class(i)
        icon = check(i)
        html += f"""
        <div class="step-node {cls}">
            <div class="step-num">{icon}</div>
            <div>
                <div class="step-label">{label}</div>
                <span class="step-sub">{sub}</span>
            </div>
        </div>"""
        if i < len(steps) - 1:
            html += '<span class="step-arrow">&#8250;</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── KPI summary ──────────────────────────────────────────────────────────────
def render_metric_summary(experiment: ExperimentResult):
    good = [m for m in experiment.model_results.values() if m.is_success]
    best_f1  = max(good, key=lambda m: m.f1_macro,  default=None)
    best_acc = max(good, key=lambda m: m.accuracy,   default=None)

    f1_val = f"{best_f1.f1_macro:.4f}" if best_f1 else "—"
    f1_name = best_f1.model_name if best_f1 else ""
    acc_val = f"{best_acc.accuracy:.4f}" if best_acc else "—"
    acc_name = best_acc.model_name if best_acc else ""

    html = f"""
    <div class="kpi-strip">
        <div class="kpi-col">
            <div class="kpi-label">Models Trained</div>
            <div class="kpi-val">{len(good)} / {len(experiment.model_results)}</div>
            <div class="kpi-sub"></div>
        </div>
        <div class="kpi-sep"></div>
        <div class="kpi-col">
            <div class="kpi-label">Best Macro F1</div>
            <div class="kpi-val">{f1_val}</div>
            <div class="kpi-sub">
                <span class="kpi-badge kpi-badge-green">↑ {f1_name}</span>
            </div>
        </div>
        <div class="kpi-sep"></div>
        <div class="kpi-col">
            <div class="kpi-label">Best Accuracy</div>
            <div class="kpi-val">{acc_val}</div>
            <div class="kpi-sub">
                <span class="kpi-badge kpi-badge-green">↑ {acc_name}</span>
            </div>
        </div>
        <div class="kpi-sep"></div>
        <div class="kpi-col">
            <div class="kpi-label">Total Time</div>
            <div class="kpi-val">{experiment.total_duration_sec:.2f} s</div>
            <div class="kpi-sub"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_class_distribution(y_train: pd.Series, y_test: Optional[pd.Series] = None, dark_mode: Optional[bool] = None) -> go.Figure:
    if dark_mode is None:
        try:
            dark_mode = bool(st.session_state.get("dark_mode", False))
        except Exception:
            dark_mode = False

    font_color = "#F8FAFC" if dark_mode else "#0F172A"
    grid_color = "#334155" if dark_mode else C_GRID

    tc = y_train.value_counts().reset_index(); tc.columns = ["Class","Count"]; tc["Set"] = "Training"
    if y_test is not None:
        te = y_test.value_counts().reset_index(); te.columns = ["Class","Count"]; te["Set"] = "Testing"
        df = pd.concat([tc, te])
    else:
        df = tc
    fig = px.bar(df, x="Class", y="Count", color="Set", barmode="group",
                 color_discrete_map={"Training": "#60A5FA" if dark_mode else C_NAVY, "Testing": "#38BDF8" if dark_mode else C_BLUE}, text="Count")
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=font_color, size=12, family="sans-serif"),
        cliponaxis=False
    )
    fig.update_layout(
        font=dict(color=font_color, family="sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320,
        margin=dict(l=45, r=20, t=35, b=30),
        legend=dict(
            orientation="h", y=1.14, x=0.5, xanchor="center",
            font=dict(color=font_color, size=11),
            title=None
        ),
        xaxis=dict(
            showgrid=False, title=None,
            tickfont=dict(color=font_color, size=12, family="sans-serif"),
            linecolor="#475569" if dark_mode else "#CBD5E1"
        ),
        yaxis=dict(
            showgrid=True, gridcolor=grid_color,
            title=dict(text="Count", font=dict(color=font_color, size=12, family="sans-serif")),
            tickfont=dict(color=font_color, size=11, family="sans-serif"),
            linecolor="#475569" if dark_mode else "#CBD5E1"
        )
    )
    return fig


def plot_model_comparison(experiment: ExperimentResult, metric_keys: List[str], dark_mode: Optional[bool] = None) -> go.Figure:
    if dark_mode is None:
        try:
            dark_mode = bool(st.session_state.get("dark_mode", False))
        except Exception:
            dark_mode = False

    font_color = "#F8FAFC" if dark_mode else "#0F172A"
    grid_color = "#334155" if dark_mode else C_GRID

    labels = {"accuracy":"Accuracy","balanced_accuracy":"Bal. Acc","f1_macro":"Macro F1",
              "cohen_kappa":"Kappa","precision_macro":"Precision","recall_macro":"Recall"}
    rows = []
    for r in experiment.model_results.values():
        if r.is_success:
            for k in metric_keys:
                rows.append({"Model": r.model_name, "Metric": labels.get(k,k), "Score": getattr(r,k,0)})
    if not rows: return go.Figure()
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Model", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=[C_BLUE, "#818CF8", C_VIOLET, C_AMBER, C_SKY, C_EMERALD],
                 text=df["Score"].apply(lambda v: f"{v:.3f}"))
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=font_color, size=11, family="sans-serif"),
        cliponaxis=False
    )
    fig.update_layout(
        font=dict(color=font_color, family="sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=360,
        margin=dict(l=45, r=20, t=35, b=20),
        yaxis=dict(
            range=[0,1.15], showgrid=True, gridcolor=grid_color,
            title=dict(text="Score", font=dict(color=font_color, size=12)),
            tickfont=dict(color=font_color, size=11)
        ),
        xaxis=dict(
            showgrid=False, title=None,
            tickfont=dict(color=font_color, size=12)
        ),
        legend=dict(
            orientation="h", y=1.12, x=.5, xanchor="center",
            font=dict(color=font_color, size=11)
        )
    )
    return fig


def plot_confusion_matrix(result: EvaluationResult, dark_mode: Optional[bool] = None) -> go.Figure:
    if dark_mode is None:
        try:
            dark_mode = bool(st.session_state.get("dark_mode", False))
        except Exception:
            dark_mode = False

    font_color = "#F8FAFC" if dark_mode else "#0F172A"
    tick_color = "#E2E8F0" if dark_mode else "#0F172A"

    cm = np.array(result.confusion_matrix)
    classes = [str(c) for c in result.classes]
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    # Clean text: Primary count bold on line 1, percentage cleanly formatted on line 2 (NO raw HTML tags)
    text = [[f"<b>{int(cm[i,j])}</b><br>({pct[i,j]:.1f}%)"
             for j in range(len(classes))] for i in range(len(classes))]

    colorscale = [[0, "#1E293B"], [0.5, "#2563EB"], [1, "#60A5FA"]] if dark_mode else [[0, "#F8FAFC"], [0.5, "#93C5FD"], [1, "#1D4ED8"]]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=13, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=tick_color, size=11),
            title=dict(text="Trials", font=dict(color=font_color, size=11)),
            thickness=14,
            xpad=4
        )
    ))
    fig.update_layout(
        font=dict(color=font_color, family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=0, r=0, t=15, b=40),
        xaxis=dict(
            title=dict(text="Predicted Class", font=dict(color=font_color, size=13)),
            tickfont=dict(color=tick_color, size=11),
            automargin=True
        ),
        yaxis=dict(
            title=dict(text="True Class", font=dict(color=font_color, size=13)),
            tickfont=dict(color=tick_color, size=11),
            autorange="reversed",
            automargin=True,
            title_standoff=2
        )
    )
    return fig


def plot_feature_importance(result: EvaluationResult, top_n: int = 15, dark_mode: Optional[bool] = None) -> go.Figure:
    if dark_mode is None:
        try:
            dark_mode = bool(st.session_state.get("dark_mode", False))
        except Exception:
            dark_mode = False

    font_color = "#F8FAFC" if dark_mode else "#0F172A"
    grid_color = "#334155" if dark_mode else C_GRID

    if not result.feature_importances: return go.Figure()
    items = list(result.feature_importances.items())[:top_n]
    feats  = [i[0] for i in reversed(items)]
    scores = [i[1] for i in reversed(items)]
    colors = [C_BLUE if s > 0.07 else (C_VIOLET if s > 0.04 else "#94A3B8") for s in scores]
    fig = go.Figure(go.Bar(x=scores, y=feats, orientation="h",
                           marker=dict(color=colors),
                           text=[f"{s:.4f}" for s in scores], textposition="outside",
                           textfont=dict(color=font_color, size=11)))
    fig.update_layout(
        font=dict(color=font_color, family="sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=max(240, len(feats)*22),
        margin=dict(l=70, r=40, t=10, b=20),
        xaxis=dict(
            showgrid=True, gridcolor=grid_color,
            title=dict(text="Gini Importance", font=dict(color=font_color, size=12)),
            tickfont=dict(color=font_color, size=11)
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(color=font_color, size=11),
            automargin=True
        )
    )
    return fig


def plot_per_class_metrics(result: EvaluationResult) -> go.Figure:
    """
    Renders grouped horizontal or vertical bar chart comparing Precision,
    Recall, and F1-Score across all classes for a specific classifier.
    """
    if not result.per_class_metrics:
        return go.Figure()

    classes = list(result.per_class_metrics.keys())
    precisions = [result.per_class_metrics[c].get("precision", 0) for c in classes]
    recalls = [result.per_class_metrics[c].get("recall", 0) for c in classes]
    f1s = [result.per_class_metrics[c].get("f1_score", 0) for c in classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=classes, y=precisions, marker_color="#2563EB", text=[f"{v:.3f}" for v in precisions], textposition="outside"))
    fig.add_trace(go.Bar(name="Recall", x=classes, y=recalls, marker_color="#7C3AED", text=[f"{v:.3f}" for v in recalls], textposition="outside"))
    fig.add_trace(go.Bar(name="F1-Score", x=classes, y=f1s, marker_color="#059669", text=[f"{v:.3f}" for v in f1s], textposition="outside"))

    fig.update_layout(
        barmode="group",
        font=dict(color="#0F172A", family="sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=20, t=35, b=50),
        yaxis=dict(
            range=[0, 1.18],
            showgrid=True,
            gridcolor=C_GRID,
            title=dict(text="Score", font=dict(color="#0F172A", size=12)),
            tickfont=dict(color="#0F172A", size=11)
        ),
        xaxis=dict(
            title=dict(text="Class", font=dict(color="#0F172A", size=12)),
            tickfont=dict(color="#0F172A", size=11)
        ),
        legend=dict(
            orientation="h",
            y=1.14,
            x=0.5,
            xanchor="center",
            font=dict(color="#0F172A", size=11)
        )
    )
    return fig


def plot_feature_distribution(
    X: pd.DataFrame,
    y: pd.Series,
    feature_name: str,
    chart_type: str = "histogram"
) -> go.Figure:
    """
    Renders an overlaid distribution or box/violin plot showing the separation
    of a selected EEG feature across brain state classes.
    """
    if feature_name not in X.columns:
        return go.Figure()

    df = pd.DataFrame({
        feature_name: X[feature_name],
        "Class": y.astype(str)
    })

    palette = [C_BLUE, C_VIOLET, C_AMBER, C_EMERALD, C_SKY]
    ctype = chart_type.lower()

    if "box" in ctype:
        fig = px.box(
            df,
            x="Class",
            y=feature_name,
            color="Class",
            points="outliers",
            color_discrete_sequence=palette
        )
        x_title = "Brain State Class"
        y_title = f"{feature_name} (Amplitude / Bandpower)"
    elif "violin" in ctype:
        fig = px.violin(
            df,
            x="Class",
            y=feature_name,
            color="Class",
            box=True,
            points=False,
            color_discrete_sequence=palette
        )
        x_title = "Brain State Class"
        y_title = f"{feature_name} (Amplitude / Bandpower)"
    else:
        fig = px.histogram(
            df,
            x=feature_name,
            color="Class",
            barmode="overlay",
            opacity=0.62,
            nbins=35,
            color_discrete_sequence=palette
        )
        fig.update_traces(
            hovertemplate="<b>%{data.name}</b><br>Value: %{x:.2f}<br>Count: %{y}<extra></extra>"
        )
        x_title = f"{feature_name} (Bandpower / Amplitude)"
        y_title = "Trial Count"

    fig.update_layout(
        font=dict(color="#0F172A", family="sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=45, r=20, t=35, b=40),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=C_GRID,
            title=dict(text=y_title, font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10)
        ),
        legend=dict(
            orientation="h",
            y=1.14,
            x=0.5,
            xanchor="center",
            font=dict(color="#0F172A", size=11),
            title=None
        )
    )
    return fig


def render_educational_drawer():
    with st.expander("📚 What do these metrics mean?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**🎯 Accuracy**
Percentage of trials classified correctly.
> Can be misleading if one class has far more samples.

**⚖️ Balanced Accuracy**
Average recall across all classes — fairer for imbalanced data.

**📐 Macro F1**
Precision × Recall balance, averaged equally per class.
> Best single metric for BCI — treats rare commands fairly.
""")
        with col2:
            st.markdown("""
**🤝 Cohen's Kappa (κ)**
Agreement corrected for chance. κ > 0.6 = acceptable BCI control.

**🔲 Confusion Matrix**
Shows exactly which brain states were confused with which.

**📊 Feature Importance**
Which EEG bands/electrodes mattered most (Random Forest only).
""")
