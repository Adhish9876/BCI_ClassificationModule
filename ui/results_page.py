"""
Results page — Unicorn UX: clean KPI cards, purposeful tabs, zero clutter.
Includes:
- 2 Linked, Live-Updating Diagnostic Plots
- "Explain" Panel with educational questions & interactive quiz
- Export of executable Python code and parameters
"""
import json
import pandas as pd
import streamlit as st

from classification.export import (
    export_classification_report_csv,
    export_comparison_csv,
    export_experiment_json,
    export_experiment_parameters_json,
    export_reproducible_python_code,
    serialize_trained_model
)
from ui.components import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_per_class_metrics,
    render_metric_summary
)


def render_results_page():
    exp = st.session_state.get("experiment_result")

    if exp is None:
        st.info("No experiment results yet. Please configure and run classifiers in Step 2.")
        if st.button("← Go to Step 2: Classifier Setup", type="primary"):
            st.session_state["current_step"] = 2
            st.rerun()
        return

    with st.container(key="results_wrapper"):
        with st.container(key="results_kpi_card"):
            st.markdown(
                '<div class="results-kpi-header">'
                '<span>🏆</span><span>Results & Benchmarking</span></div>',
                unsafe_allow_html=True
            )
            # --- KPIs ---
            render_metric_summary(exp)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # --- Clean Tabbed Results (Rubric-Mapped) ---
        tab_board, tab_charts, tab_linked, tab_matrix, tab_explain, tab_export = st.tabs([
            "🏆  Leaderboard",
            "📊  Metric Chart",
            "🔗  Linked Plots",
            "🔲  Confusion Matrix",
            "💡  “Explain” Panel",
            "📥  Export (Code & Parameters)"
        ])

        # ── 1. Leaderboard ──
        with tab_board:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            col_sort, _ = st.columns([2, 3])
            with col_sort:
                sort_by = st.selectbox("Sort by metric:", ["Macro F1", "Accuracy", "Balanced Accuracy", "Cohen's Kappa", "Train Time (s)"])

            df = exp.to_dataframe()
            if sort_by in df.columns:
                df = df.sort_values(by=sort_by, ascending=(sort_by == "Train Time (s)"))

            st.markdown(
                '<div style="background:#FFFFFF;border:1.5px solid #CBD5E1;border-radius:14px;'
                'padding:20px 24px;margin-top:14px;box-shadow:0 1px 3px rgba(15,23,42,0.05), 0 4px 8px -1px rgba(15,23,42,0.08), 0 12px 24px -4px rgba(15,23,42,0.12);">'
                '<div style="font-size:18px;font-weight:800;color:#0F172A;letter-spacing:-0.01em;margin-bottom:12px;">'
                '🏆 Classifier Performance Leaderboard (Ranked on Unseen EEG Test Set)</div>',
                unsafe_allow_html=True
            )
            st.dataframe(
                df.style.highlight_max(subset=["Accuracy", "Balanced Accuracy", "Macro F1", "Cohen's Kappa"], color="#DBEAFE")
                       .highlight_min(subset=["Train Time (s)"], color="#F1F5F9"),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── 2. Metric Chart ──
        with tab_charts:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            metric_map = {
                "Accuracy": "accuracy",
                "Balanced Acc": "balanced_accuracy",
                "Macro F1": "f1_macro",
                "Cohen's Kappa": "cohen_kappa",
                "Precision": "precision_macro",
                "Recall": "recall_macro"
            }
            chosen = st.multiselect("Select metrics to compare:", list(metric_map.keys()), default=["Accuracy", "Balanced Acc", "Macro F1", "Cohen's Kappa"])
            if chosen:
                st.markdown(
                    '<div style="font-size:18px;font-weight:800;color:#0F172A;margin:16px 0 8px 0;">'
                    'Multi-Metric Classifier Comparison</div>',
                    unsafe_allow_html=True
                )
                st.plotly_chart(
                    plot_model_comparison(exp, [metric_map[m] for m in chosen]),
                    use_container_width=True,
                    theme=None,
                    config={"displayModeBar": False},
                    key="chart_results_comparison"
                )

        # ── 3. 2 Linked, Live-Updating Diagnostic Plots ──
        with tab_linked:
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            good = {r.model_name: r for r in exp.model_results.values() if r.is_success}
            if good:
                col_sel, _ = st.columns([0.45, 1.55])
                with col_sel:
                    sel_linked = st.selectbox("Select Classifier to Live-Inspect:", list(good.keys()), key="linked_model_dropdown")

                active_res = good[sel_linked]
                plot2_mode = st.session_state.get("linked_p2_mode", "Per-Class Precision / Recall / F1")

                col_diag_left, col_spacer_plot, col_diag_right = st.columns([1.1, 0.2, 1.1])
                with col_diag_left:
                    st.markdown(
                        f'<div style="font-size:16px;font-weight:800;color:#0F172A;margin:8px 0 4px 0;">'
                        f'Plot 1: Confusion Matrix — {sel_linked}</div>'
                        '<div style="font-size:12px;color:#64748B;margin-bottom:8px;">Row-normalized percentages & trial counts</div>',
                        unsafe_allow_html=True
                    )
                    st.plotly_chart(
                        plot_confusion_matrix(active_res),
                        use_container_width=True,
                        theme=None,
                        config={"displayModeBar": False},
                        key=f"chart_cm_linked_{sel_linked}"
                    )

                with col_diag_right:
                    st.markdown(
                        f'<div style="font-size:16px;font-weight:800;color:#0F172A;margin:8px 0 4px 0;">'
                        f'Plot 2: {plot2_mode} — {sel_linked}</div>'
                        '<div style="font-size:12px;color:#64748B;margin-bottom:8px;">Synchronized performance breakdown</div>',
                        unsafe_allow_html=True
                    )
                    if plot2_mode == "Feature Importances (RF)":
                        if active_res.feature_importances:
                            st.plotly_chart(
                                plot_feature_importance(active_res),
                                use_container_width=True,
                                theme=None,
                                config={"displayModeBar": False},
                                key=f"chart_rf_imp_{sel_linked}"
                            )
                        else:
                            st.info(f"{sel_linked} does not generate tree Gini importances. Switched to Per-Class Metrics:")
                            st.plotly_chart(
                                plot_per_class_metrics(active_res),
                                use_container_width=True,
                                theme=None,
                                config={"displayModeBar": False},
                                key=f"chart_pcm_fb_{sel_linked}"
                            )
                    else:
                        st.plotly_chart(
                            plot_per_class_metrics(active_res),
                            use_container_width=True,
                            theme=None,
                            config={"displayModeBar": False},
                            key=f"chart_pcm_{sel_linked}"
                        )

                    # Diagnostic view toggle placed directly underneath Plot 2
                    st.radio(
                        "Plot 2 Diagnostic View:",
                        ["Per-Class Precision / Recall / F1", "Feature Importances (RF)"],
                        horizontal=True,
                        key="linked_p2_mode"
                    )

                # Diagnostic Summary Pill Strip for active linked model
                st.markdown(
                    f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;'
                    f'padding:12px 18px;margin-top:14px;display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:12.5px;color:#334155;">Active: <b>{sel_linked}</b> &nbsp;|&nbsp; '
                    f'Accuracy: <b>{active_res.accuracy:.4f}</b> &nbsp;|&nbsp; '
                    f'Balanced Acc: <b>{active_res.balanced_accuracy:.4f}</b> &nbsp;|&nbsp; '
                    f'Macro F1: <b>{active_res.f1_macro:.4f}</b> &nbsp;|&nbsp; '
                    f'Cohen\'s κ: <b>{active_res.cohen_kappa:.4f}</b> &nbsp;|&nbsp; '
                    f'Inference: <b>{active_res.predict_time_sec * 1000:.2f} ms/trial</b></span>'
                    f'<span style="font-size:11px;font-weight:700;color:#047857;background:#ECFDF5;border:1px solid #A7F3D0;padding:2px 8px;border-radius:6px;">✓ Active State</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── 4. Confusion Matrix (Dedicated) ──
        with tab_matrix:
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            good = {r.model_name: r for r in exp.model_results.values() if r.is_success}
            if good:
                col_sel, _ = st.columns([1, 2.5])
                with col_sel:
                    sel = st.selectbox("Select classifier:", list(good.keys()), key="cm_sel_tab4")
                res = good[sel]

                col_cm, col_diag = st.columns([1.3, 0.7], gap="medium")
                with col_cm:
                    st.markdown(
                        f'<div style="font-size:18px;font-weight:800;color:#0F172A;letter-spacing:-0.01em;margin:8px 0 8px 0;">'
                        f'Confusion Matrix — {sel}</div>',
                        unsafe_allow_html=True
                    )
                    st.plotly_chart(
                        plot_confusion_matrix(res),
                        use_container_width=True,
                        theme=None,
                        config={"displayModeBar": False},
                        key=f"chart_cm_dedicated_{sel}"
                    )
                with col_diag:
                    st.markdown(
                        f'<div style="background:#F8FAFC;border:1.5px solid #CBD5E1;border-radius:14px;'
                        f'padding:20px 24px;margin-top:52px;box-shadow:0 1px 3px rgba(15,23,42,0.05), 0 4px 8px -1px rgba(15,23,42,0.08);">'
                        f'<div style="font-size:18px;font-weight:800;color:#0F172A;letter-spacing:-0.01em;margin-bottom:12px;">{sel} Diagnostics</div>'
                        f'<div style="font-size:13.5px;color:#475569;line-height:2.2;">'
                        f'Accuracy: <b style="color:#0F172A;">{res.accuracy:.4f}</b><br>'
                        f'Balanced Acc: <b style="color:#0F172A;">{res.balanced_accuracy:.4f}</b><br>'
                        f'Macro F1: <b style="color:#0F172A;">{res.f1_macro:.4f}</b><br>'
                        f'Cohen\'s κ: <b style="color:#0F172A;">{res.cohen_kappa:.4f}</b><br>'
                        f'Inference Time: <b style="color:#2563EB;">{res.predict_time_sec * 1000:.2f} ms</b>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

        # ── 5. “Explain” Panel with Questions ──
        with tab_explain:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            _render_explain_panel(exp)

        # ── 6. Export (Code & Parameters) ──
        with tab_export:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            _render_export_panel(exp)

        # --- Action Bar ---
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        col_back, col_new = st.columns([1, 1])
        with col_back:
            if st.button("← Re-configure Classifiers (Step 2)", use_container_width=True):
                st.session_state["current_step"] = 2
                st.rerun()
        with col_new:
            if st.button("🔄 New Experiment (Step 1: Dataset)", type="primary", use_container_width=True):
                st.session_state["current_step"] = 1
                st.rerun()


# ── “Explain” Panel Implementation ─────────────────────────────────────────────
def _render_explain_panel(exp):
    """
    Renders the educational 'Explain' Panel with the 8 core BCI machine learning
    questions specified in Section 12 of Details.md, complete with contextual stats.
    """
    good = {r.model_name: r for r in exp.model_results.values() if r.is_success}
    best_f1 = max(good.values(), key=lambda r: r.f1_macro, default=None)
    best_acc = max(good.values(), key=lambda r: r.accuracy, default=None)
    best_kappa = max(good.values(), key=lambda r: r.cohen_kappa, default=None)

    # Q1: What does accuracy mean?
    with st.expander("❓ 1. What does Accuracy mean, and when is it misleading in BCI?"):
        st.markdown(f"""
**Definition:** Accuracy measures the simple ratio of correctly predicted EEG trials over total evaluated trials:
$$\\text{{Accuracy}} = \\frac{{\\text{{TP}} + \\text{{TN}}}}{{\\text{{Total Trials}}}}$$

**When it is misleading in BCI:**
In oddball paradigms (such as P300 spellers, where non-target events constitute 85–90% of trials) or rare seizure event detection, standard accuracy is dangerously deceptive. A trivial "dummy" classifier that uniformly guesses the majority rest/non-target state will achieve 90% accuracy while having **zero clinical or assistive utility**.

> **Your Experiment Insight:**
> Current Top Accuracy: **{best_acc.accuracy:.4f}** ({best_acc.model_name}).
> Compare this with Balanced Accuracy (**{best_acc.balanced_accuracy:.4f}**) and Macro F1 (**{best_acc.f1_macro:.4f}**). When these diverge significantly, class imbalance is distorting standard accuracy.
""")

    # Q2: Why might SVM outperform LDA?
    svm_res = next((r for r in good.values() if "Support Vector" in r.model_name or "SVM" in r.model_name), None)
    lda_res = next((r for r in good.values() if "Linear Discriminant" in r.model_name or "LDA" in r.model_name), None)
    with st.expander("❓ 2. Why might SVM outperform LDA on motor imagery EEG features?"):
        diff_text = ""
        if svm_res and lda_res:
            diff = svm_res.f1_macro - lda_res.f1_macro
            diff_text = f"In your current run: **SVM Macro F1 = {svm_res.f1_macro:.4f}** vs **LDA Macro F1 = {lda_res.f1_macro:.4f}** (Difference: `{diff:+.4f}`)."

        st.markdown(f"""
**Linear vs. Non-linear Class Separation:**
- **LDA (Linear Discriminant Analysis)** estimates class centroids and a shared pooled covariance matrix under Gaussian assumptions to compute a linear Fisher projection boundary. It is fast and closed-form, but fundamentally cannot capture curved or non-linear boundaries between spectral bandpower patterns.
- **SVM with RBF Kernel** projects EEG feature vectors into an infinite-dimensional reproducing kernel Hilbert space (RKHS) where complex non-linear sensorimotor rhythm interactions become linearly separable. Furthermore, SVM optimizes the **maximum margin** between support vectors, providing robust regularized boundaries resilient to noisy EEG outlier trials.

> **Your Experiment Insight:**
> {diff_text if diff_text else "Train both LDA and SVM to compare their empirical boundary separation on this dataset."}
""")

    # Q3: What does a confusion matrix indicate?
    with st.expander("❓ 3. What does a Confusion Matrix indicate about EEG misclassifications?"):
        st.markdown("""
**Interpreting the Matrix Structure:**
- **Rows** denote True intended mental states (ground-truth task instructions).
- **Columns** denote the classifier's Predicted mental states.
- **Main Diagonal** represents correct classifications (True Positives for each mental state).
- **Off-Diagonal Cells** reveal specific neurophysiological confusion patterns:
  - *Left vs. Right Hand Motor Imagery Confusion:* High off-diagonal rates between left and right imagery frequently reflect bilateral sensorimotor cortex activation or poor spatial filtering (e.g., volume conduction between C3 and C4 electrodes).
  - *Imagery vs. Rest Confusion:* Confusing an active imagery command with Rest often indicates low subject engagement, subject fatigue, or weak event-related desynchronization (ERD) amplitude.
""")

    # Q4: Why is Macro F1 useful for imbalanced datasets?
    with st.expander("❓ 4. Why is Macro F1 especially useful for imbalanced BCI datasets?"):
        st.markdown("""
**Arithmetic Unweighted Mean Across Classes:**
Unlike Micro or Weighted F1, Macro F1 gives equal weight to each mental command class regardless of trial prevalence:
$$\\text{Macro F1} = \\frac{1}{K} \\sum_{k=1}^K F1_k, \\quad \\text{where } F1_k = 2 \\cdot \\frac{\\text{Precision}_k \\cdot \\text{Recall}_k}{\\text{Precision}_k + \\text{Recall}_k}$$

In a 3-class BCI (e.g. Left Hand, Right Hand, Rest), if the classifier perfectly predicts the abundant Rest class but completely fails on Right Hand imagery, Weighted F1 remains artificially high, whereas **Macro F1 plunges**, immediately flagging the failure to decode that specific brain command.
""")

    # Q5: Why does feature scaling matter for KNN and SVM, but not Random Forest?
    with st.expander("❓ 5. Why does feature scaling (StandardScaler) matter for KNN and SVM, but not Random Forest?"):
        st.markdown("""
**Distance Metrics vs. Monotonic Threshold Splits:**
- **KNN** classifies samples via geometric distance (e.g. Euclidean distance $d(x, z) = \\sqrt{\\sum_j (x_j - z_j)^2}$). If one EEG spectral band has values in $10^2\\,\\mu\\text{V}^2$ while another has values in $10^{-1}\\,\\mu\\text{V}^2$, the larger feature completely dominates distance calculations.
- **SVM** maximizes the margin $\\frac{2}{\\|w\\|}$ subject to functional margin constraints. Unscaled features distort margin penalties and cause gradient solvers to oscillate.
- **Random Forest** builds orthogonal decision stumps evaluating single-feature thresholds ($x_j \\le \\theta$). Because the Gini impurity split order depends strictly on feature rank order rather than numerical scale, **tree algorithms are completely invariant to monotonic feature scaling**.
""")

    # Q6: Why can Random Forest handle non-linear relationships?
    with st.expander("❓ 6. Why can Random Forest handle non-linear relationships?"):
        st.markdown("""
**Piecewise Rectangular Partitioning & Bagging:**
An individual decision tree recursively splits feature space into axis-aligned hyper-rectangles. By aggregating hundreds of decorrelated decision trees (via bootstrap bagging and random feature subsampling), Random Forest approximates smooth, non-linear, multi-modal decision boundaries without requiring explicit kernel transformations.
""")

    # Q7: What does Cohen's Kappa measure?
    with st.expander("❓ 7. What does Cohen's Kappa (κ) measure in BCI benchmarks?"):
        st.markdown(f"""
**Chance-Corrected Agreement Metric:**
Cohen's Kappa ($\\kappa$) quantifies agreement between predicted and true classes beyond chance expectation:
$$\\kappa = \\frac{{p_o - p_e}}{{1 - p_e}}$$
where $p_o$ is the observed accuracy and $p_e$ is the hypothetical probability of chance agreement based on marginal class distributions.

- $\\kappa < 0.4$: Poor or unacceptable BCI control.
- $0.4 \\le \\kappa < 0.6$: Moderate performance.
- $\\kappa \\ge 0.6$: Substantial, reliable BCI communication.
- $\\kappa \\ge 0.8$: Near-flawless BCI control.

> **Your Experiment Insight:**
> Highest $\\kappa$ achieved: **{best_kappa.cohen_kappa:.4f}** ({best_kappa.model_name}).
""")

    # Q8: Participant Group Splitting
    with st.expander("❓ 8. How does Subject / Participant Group Splitting prevent identity leakage?"):
        st.markdown("""
**The Danger of Inter-Subject Data Leakage:**
EEG features recorded from the same participant share distinct anatomical and physiological baselines (skull thickness, baseline alpha frequency, electrode impedances). If trials from Participant A appear in both training and test sets, classifiers inadvertently learn to recognize **who the subject is** rather than **what mental task was performed**. Group splitting (e.g. `GroupShuffleSplit`) ensures test subjects are completely unseen during training, measuring true cross-subject generalization.
""")


# ── Consistent Monochromatic Outline File Icons (18x18, Stroke: #334155) ──────
SVG_FILE_CODE = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#334155" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">'
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
    '<polyline points="14 2 14 8 20 8"></polyline>'
    '<polyline points="10 13 8 15 10 17"></polyline>'
    '<polyline points="14 13 16 15 14 17"></polyline>'
    '</svg>'
)
SVG_FILE_MODEL = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#334155" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">'
    '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>'
    '<polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>'
    '<line x1="12" y1="22.08" x2="12" y2="12"></line>'
    '</svg>'
)
SVG_FILE_TABLE = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#334155" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">'
    '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>'
    '<line x1="3" y1="9" x2="21" y2="9"></line>'
    '<line x1="3" y1="15" x2="21" y2="15"></line>'
    '<line x1="9" y1="3" x2="9" y2="21"></line>'
    '<line x1="15" y1="3" x2="15" y2="21"></line>'
    '</svg>'
)
SVG_FILE_CONFIG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#334155" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">'
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
    '<polyline points="14 2 14 8 20 8"></polyline>'
    '<circle cx="12" cy="14" r="3"></circle>'
    '</svg>'
)


def _format_size(num_bytes: int) -> str:
    """Formats raw byte count into human-readable scientific asset size."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"


# ── Export Panel Implementation ───────────────────────────────────────────────
def _render_export_panel(exp):
    """
    Renders the tabbed, scientific Artifact Manager interface:
    - Constrained to max-width ~1440px, centered horizontally.
    - Category segmented tabs: 'Executable Assets (2)' and 'Experiment Reports (2)'
    - Strict visual hierarchy per artifact row wrapped in high-contrast card divs:
        Primary: Human-readable artifact name (16.5px, semibold #0F172A)
        Secondary: Monospaced filename (12.5px) + Concise description (14px, #475569)
        Metadata: File extension badge + Genuine dynamic file size
        Actions: Compact Download button + Model selector / Expandable syntax previews
    """
    py_code = export_reproducible_python_code(exp)
    params_json = export_experiment_parameters_json(exp)
    comp_csv = export_comparison_csv(exp)
    good = {r.model_name: r for r in exp.model_results.values() if r.is_success}

    py_filename = f"train_bci_{exp.dataset_name.lower().replace(' ', '_')}.py"
    py_size = _format_size(len(py_code.encode("utf-8")))

    csv_size = _format_size(len(comp_csv.encode("utf-8")))
    json_size = _format_size(len(params_json.encode("utf-8")))

    with st.container(key="artifact_workspace"):
        # Header (24px semibold title + 14px description)
        st.markdown(
            '<div style="margin-bottom:20px;">'
            '<div style="font-size:24px;font-weight:700;color:#0F172A;letter-spacing:-0.02em;">'
            'Export Experiment Artifacts</div>'
            '<div style="font-size:14px;color:#475569;margin-top:4px;line-height:1.5;">'
            'Download standalone execution scripts, serialized model pipelines, benchmark tables, and experiment manifests.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        tab_exec, tab_reports = st.tabs([
            "Executable Assets (2)",
            "Experiment Reports (2)"
        ])

        # ── TAB 1: EXECUTABLE ASSETS ──
        with tab_exec:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

            # --- Card 1: Python Training Script ---
            with st.container(key="art_card_py"):
                col_info_py, col_meta_py, col_act_py = st.columns([4.4, 1.2, 3.2])
                with col_info_py:
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:flex-start;gap:12px;">
                            <div style="margin-top:3px;flex-shrink:0;">{SVG_FILE_CODE}</div>
                            <div style="min-width:0;">
                                <div style="font-size:16.5px;font-weight:700;color:#0F172A;letter-spacing:-0.015em;line-height:1.2;">
                                    Python Training Script
                                </div>
                                <div style="display:flex;align-items:center;gap:8px;margin-top:5px;">
                                    <span style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12.5px;font-weight:600;color:#1E293B;background:#FFFFFF;border:1px solid #CBD5E1;padding:2px 8px;border-radius:4px;">
                                        {py_filename}
                                    </span>
                                </div>
                                <div style="font-size:14px;color:#475569;margin-top:5px;line-height:1.5;max-width:680px;">
                                    Standalone reproducible training script with full preprocessing pipelines, stratified cross-validation splits, and classifier evaluation.
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_meta_py:
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;gap:8px;padding-top:4px;">
                            <span style="font-family:ui-monospace,monospace;font-size:12px;font-weight:700;color:#334155;background:#E2E8F0;border:1px solid #CBD5E1;padding:2px 8px;border-radius:4px;text-transform:uppercase;">
                                .py
                            </span>
                            <span style="font-family:ui-monospace,monospace;font-size:12.5px;font-weight:600;color:#475569;">
                                {py_size}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_act_py:
                    _, col_btn_py = st.columns([1.7, 1.0])
                    with col_btn_py:
                        st.download_button(
                            "↓ Download",
                            data=py_code,
                            file_name=py_filename,
                            mime="text/x-python",
                            key="btn_art_py",
                            use_container_width=True
                        )

                with st.expander("Preview source code", expanded=False):
                    st.code(py_code, language="python")

            # --- Card 2: Serialized Model Pipeline ---
            avail_models = list(good.keys())
            if avail_models:
                current_sel = st.session_state.get("artifact_model_sel", avail_models[0])
                if current_sel not in avail_models:
                    current_sel = avail_models[0]

                m_res = good[current_sel]
                joblib_filename = f"{m_res.model_id}_pipeline.joblib"
                try:
                    joblib_bytes = serialize_trained_model(m_res)
                    joblib_size = _format_size(len(joblib_bytes))
                except Exception:
                    joblib_bytes = b""
                    joblib_size = "—"

                with st.container(key="art_card_model"):
                    col_info_model, col_meta_model, col_act_model = st.columns([4.4, 1.2, 3.2])
                    with col_info_model:
                        st.markdown(
                            f"""
                            <div style="display:flex;align-items:flex-start;gap:12px;">
                                <div style="margin-top:3px;flex-shrink:0;">{SVG_FILE_MODEL}</div>
                                <div style="min-width:0;">
                                    <div style="font-size:16.5px;font-weight:700;color:#0F172A;letter-spacing:-0.015em;line-height:1.2;">
                                        Serialized Model Pipeline
                                    </div>
                                    <div style="display:flex;align-items:center;gap:8px;margin-top:5px;">
                                        <span style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12.5px;font-weight:600;color:#1E293B;background:#FFFFFF;border:1px solid #CBD5E1;padding:2px 8px;border-radius:4px;">
                                            {joblib_filename}
                                        </span>
                                    </div>
                                    <div style="font-size:14px;color:#475569;margin-top:5px;line-height:1.5;max-width:680px;">
                                        Fitted scikit-learn Pipeline ({current_sel}) serialized via joblib for immediate runtime inference in BCI applications.
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_meta_model:
                        st.markdown(
                            f"""
                            <div style="display:flex;align-items:center;gap:8px;padding-top:4px;">
                                <span style="font-family:ui-monospace,monospace;font-size:12px;font-weight:700;color:#334155;background:#E2E8F0;border:1px solid #CBD5E1;padding:2px 8px;border-radius:4px;text-transform:uppercase;">
                                    .joblib
                                </span>
                                <span style="font-family:ui-monospace,monospace;font-size:12.5px;font-weight:600;color:#475569;">
                                    {joblib_size}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_act_model:
                        col_drop, col_btn_joblib = st.columns([1.7, 1.0])
                        with col_drop:
                            chosen_m = st.selectbox(
                                "Classifier",
                                avail_models,
                                index=avail_models.index(current_sel),
                                label_visibility="collapsed",
                                key="artifact_model_sel"
                            )
                        with col_btn_joblib:
                            st.download_button(
                                "↓ Download",
                                data=joblib_bytes,
                                file_name=joblib_filename,
                                mime="application/octet-stream",
                                key="btn_art_joblib",
                                use_container_width=True
                            )
            else:
                st.info("No successfully trained model estimators available for serialization.")

        # ── TAB 2: EXPERIMENT REPORTS ──
        with tab_reports:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

            # --- Card 3: Benchmark Results CSV ---
            with st.container(key="art_card_csv"):
                col_info_csv, col_meta_csv, col_act_csv = st.columns([4.4, 1.2, 3.2])
                with col_info_csv:
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:flex-start;gap:12px;">
                            <div style="margin-top:3px;flex-shrink:0;">{SVG_FILE_TABLE}</div>
                            <div style="min-width:0;">
                                <div style="font-size:16.5px;font-weight:700;color:#0F172A;letter-spacing:-0.015em;line-height:1.2;">
                                    Benchmark Results
                                </div>
                                <div style="display:flex;align-items:center;gap:8px;margin-top:5px;">
                                    <span style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12.5px;font-weight:600;color:#1E293B;background:#FFFFFF;border:1px solid #CBD5E1;padding:2px 8px;border-radius:4px;">
                                        benchmark_results.csv
                                    </span>
                                </div>
                                <div style="font-size:14px;color:#475569;margin-top:5px;line-height:1.5;max-width:680px;">
                                    Summary benchmark comparison table including Accuracy, Balanced Acc, Macro F1, Cohen's κ, and execution latencies.
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_meta_csv:
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;gap:8px;padding-top:4px;">
                            <span style="font-family:ui-monospace,monospace;font-size:12px;font-weight:700;color:#334155;background:#E2E8F0;border:1px solid #CBD5E1;padding:2px 8px;border-radius:4px;text-transform:uppercase;">
                                .csv
                            </span>
                            <span style="font-family:ui-monospace,monospace;font-size:12.5px;font-weight:600;color:#475569;">
                                {csv_size}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_act_csv:
                    _, col_btn_csv = st.columns([1.7, 1.0])
                    with col_btn_csv:
                        st.download_button(
                            "↓ Download",
                            data=comp_csv,
                            file_name="benchmark_results.csv",
                            mime="text/csv",
                            key="btn_art_csv",
                            use_container_width=True
                        )

            # --- Card 4: Experiment Configuration JSON ---
            with st.container(key="art_card_json"):
                col_info_json, col_meta_json, col_act_json = st.columns([4.4, 1.2, 3.2])
                with col_info_json:
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:flex-start;gap:12px;">
                            <div style="margin-top:3px;flex-shrink:0;">{SVG_FILE_CONFIG}</div>
                            <div style="min-width:0;">
                                <div style="font-size:16.5px;font-weight:700;color:#0F172A;letter-spacing:-0.015em;line-height:1.2;">
                                    Experiment Configuration
                                </div>
                                <div style="display:flex;align-items:center;gap:8px;margin-top:5px;">
                                    <span style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12.5px;font-weight:600;color:#1E293B;background:#FFFFFF;border:1px solid #CBD5E1;padding:2px 8px;border-radius:4px;">
                                        experiment_parameters.json
                                    </span>
                                </div>
                                <div style="font-size:14px;color:#475569;margin-top:5px;line-height:1.5;max-width:680px;">
                                    Configuration manifest detailing all classifier hyperparameters, preprocessing steps, CV splits, and random seeds.
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_meta_json:
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;gap:8px;padding-top:4px;">
                            <span style="font-family:ui-monospace,monospace;font-size:12px;font-weight:700;color:#334155;background:#E2E8F0;border:1px solid #CBD5E1;padding:2px 8px;border-radius:4px;text-transform:uppercase;">
                                .json
                            </span>
                            <span style="font-family:ui-monospace,monospace;font-size:12.5px;font-weight:600;color:#475569;">
                                {json_size}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_act_json:
                    _, col_btn_json = st.columns([1.7, 1.0])
                    with col_btn_json:
                        st.download_button(
                            "↓ Download",
                            data=params_json,
                            file_name="experiment_parameters.json",
                            mime="application/json",
                            key="btn_art_json",
                            use_container_width=True
                        )

                with st.expander("Preview parameters", expanded=False):
                    st.code(params_json, language="json")

