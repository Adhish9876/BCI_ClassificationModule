"""
Model page — Playing card grid layout with tags beside names and ⚙️ icon.
Linear/Stripe inspired high-depth cards. Supports Light and Dark modes.
"""
import time
import streamlit as st
from classification.model_registry import get_registered_models
from classification.training import train_and_evaluate


MODEL_CARD_INFO = {
    "lda": {
        "short_name": "LDA",
        "full_name": "Linear Discriminant Analysis",
        "tag": "Linear Discriminative",
        "meta": "Linear · Statistical",
        "desc": "Finds linear combination of features maximizing separation between class centroids relative to within-class spread.",
    },
    "svm": {
        "short_name": "SVM",
        "full_name": "Support Vector Machine",
        "tag": "Kernel / Max Margin",
        "meta": "Kernel · Margin-based",
        "desc": "Finds optimal separating hyperplane maximizing geometric margin to nearest support vectors.",
    },
    "rf": {
        "short_name": "Random Forest",
        "full_name": "Random Forest Classifier",
        "tag": "Ensemble Trees",
        "meta": "Ensemble · Non-linear",
        "desc": "Builds decorrelated decision trees using bootstrap aggregation and random feature subsets.",
    },
    "knn": {
        "short_name": "KNN",
        "full_name": "K-Nearest Neighbors",
        "tag": "Instance-Based",
        "meta": "Instance-based · Non-parametric",
        "desc": "Classifies unseen test samples based on majority class labels among their K closest training neighbors.",
    },
    "lr": {
        "short_name": "Logistic Regression",
        "full_name": "Logit / Sigmoid Linear",
        "tag": "Probabilistic Linear",
        "meta": "Linear · Probabilistic",
        "desc": "Models posterior class probabilities via logistic sigmoid transformation of linear feature combinations.",
    },
    "riemann_mdm": {
        "short_name": "Riemannian MDM",
        "full_name": "Minimum Distance to Mean",
        "tag": "Riemannian Geometry",
        "meta": "SPD Manifold · Covariance",
        "desc": "Classifies covariance matrices directly on Riemannian manifold of Symmetric Positive Definite matrices.",
    },
}


def render_model_page():
    bundle = st.session_state.get("dataset_bundle")
    if bundle is None:
        st.warning("No dataset loaded.")
        if st.button("← Back to Step 1", type="primary"):
            st.session_state["current_step"] = 1
            st.rerun()
        return

    dark_mode = st.session_state.get("dark_mode", False)
    name_color = "#F8FAFC" if dark_mode else "#0F172A"
    sub_color = "#94A3B8" if dark_mode else "#64748B"

    registered = get_registered_models(include_optional=True)
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = ["lda", "svm", "rf"]
    if "model_configs" not in st.session_state:
        st.session_state["model_configs"] = {}
    if "pipeline_params" not in st.session_state:
        st.session_state["pipeline_params"] = {
            "scaler": "standard",
            "cv_folds": 5,
            "random_seed": 42,
            "primary_metric": "Macro F1-Score",
            "class_weight": None,
        }

    # ── Global Pipeline & Evaluation Parameters Card (Less White Background) ──
    st.markdown("""
    <style>
    /* Pipeline parameters container - less white background */
    div[class*="st-key-pipeline_params_card"] {
        background: #F1F5F9 !important;
        background-color: #F1F5F9 !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 22px 24px !important;
        margin-top: 6px !important;
        margin-bottom: 26px !important;
        box-shadow: 
            0 1px 3px rgba(15, 23, 42, 0.05),
            0 4px 8px -1px rgba(15, 23, 42, 0.08),
            0 12px 24px -4px rgba(15, 23, 42, 0.06) !important;
    }
    div[class*="st-key-pipeline_params_card"]:empty,
    div[class*="st-key-pipeline_params_card"]:not(:has(div[data-testid="stElementContainer"])) {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        border: none !important;
        margin: 0 !important;
    }
    div[class*="st-key-pipeline_params_card"] div[data-testid="stVerticalBlock"],
    div[class*="st-key-pipeline_params_card"] div[data-testid="stHorizontalBlock"],
    div[class*="st-key-pipeline_params_card"] div[data-testid="stElementContainer"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    div[class*="st-key-pipeline_params_card"] label p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
    }

    /* Selected model card - clean green border with reduced subtle shadow (no heavy bloom) */
    div[class*="st-key-card_"]:has(input:checked),
    div[class*="st-key-card_"]:has([aria-checked="true"]) {
        border: 2px solid #10B981 !important;
        border-color: #10B981 !important;
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 4px 12px -2px rgba(15, 23, 42, 0.08) !important;
        transform: translateY(-2px) !important;
    }
    div[class*="st-key-card_"] [data-testid="stCheckbox"] label,
    div[class*="st-key-card_"] [data-testid="stCheckbox"] label div {
        background: transparent !important;
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container(border=False, key="pipeline_params_card"):
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            scaler_map = {
                "StandardScaler (Zero Mean, Unit Var)": "standard",
                "RobustScaler (Median / IQR Outliers)": "robust",
                "MinMaxScaler (0 to 1 Range)": "minmax",
                "None (Raw Features)": "none"
            }
            inv_scaler = {v: k for k, v in scaler_map.items()}
            curr_scaler_label = inv_scaler.get(st.session_state["pipeline_params"].get("scaler", "standard"), "StandardScaler (Zero Mean, Unit Var)")
            scaler_opts = list(scaler_map.keys())
            scaler_idx = scaler_opts.index(curr_scaler_label) if curr_scaler_label in scaler_opts else 0

            scaler_opt = st.selectbox(
                "1. Feature Scaler",
                scaler_opts,
                index=scaler_idx,
                key="global_scaler_sel",
                help="Scaling ensures distance/margin-based classifiers (KNN, SVM, LR) are not dominated by high-variance channels."
            )
            st.session_state["pipeline_params"]["scaler"] = scaler_map[scaler_opt]

        with col_g2:
            cv_k = st.slider(
                "2. Cross-Validation Folds (K)",
                min_value=3,
                max_value=10,
                value=int(st.session_state["pipeline_params"].get("cv_folds", 5)),
                step=1,
                key="global_cv_slider",
                help="Stratified K-Fold partition size for internal cross-validation and variance estimation."
            )
            st.session_state["pipeline_params"]["cv_folds"] = cv_k

        with col_g3:
            seed_val = st.number_input(
                "3. Random Seed (Reproducibility)",
                min_value=0,
                max_value=9999,
                value=int(st.session_state["pipeline_params"].get("random_seed", 42)),
                step=1,
                key="global_seed_input",
                help="Controls stochastic components (Random Forest bagging, SVM dual solver, CV splitting) for deterministic reproduction."
            )
            st.session_state["pipeline_params"]["random_seed"] = int(seed_val)

        col_g4, col_g5 = st.columns(2)
        with col_g4:
            metric_opts = ["Macro F1-Score", "Balanced Accuracy", "Accuracy", "Cohen's Kappa (κ)"]
            curr_metric = st.session_state["pipeline_params"].get("primary_metric", "Macro F1-Score")
            m_idx = metric_opts.index(curr_metric) if curr_metric in metric_opts else 0
            metric_opt = st.selectbox(
                "4. Primary Benchmark Metric",
                metric_opts,
                index=m_idx,
                key="global_metric_sel",
                help="Target objective metric used to rank classifiers on the test leaderboard."
            )
            st.session_state["pipeline_params"]["primary_metric"] = metric_opt

        with col_g5:
            curr_weight = st.session_state["pipeline_params"].get("class_weight")
            w_idx = 1 if curr_weight == "balanced" else 0
            weight_opt = st.selectbox(
                "5. Class Imbalance Strategy",
                ["Standard (Uniform Weights)", "Balanced (Inverse Class Frequency)"],
                index=w_idx,
                key="global_weight_sel",
                help="'Balanced' automatically weights loss inversely proportional to class frequencies to prevent majority-class bias."
            )
            st.session_state["pipeline_params"]["class_weight"] = "balanced" if "Balanced" in weight_opt else None

    # Preset header (Select All removed per user request)
    col_hdr, col_p1 = st.columns([3, 1.2])
    with col_hdr:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0 0 0;">'
            f'<span style="font-size:21px;font-weight:800;color:{name_color};letter-spacing:-0.02em;">Classification Models</span>'
            f'<span style="background:#ECFDF5;color:#047857;border:1px solid #A7F3D0;font-size:12px;'
            f'font-weight:700;padding:2px 9px;border-radius:10px;">{len(registered)} available</span>'
            f'</div>'
            f'<div style="font-size:14px;color:{sub_color};margin-top:3px;">Select models to include in the evaluation. Hyperparameters accessible via ⚙️.</div>',
            unsafe_allow_html=True
        )
    with col_p1:
        if st.button("Recommended (LDA+SVM+RF)", key="btn_rec", use_container_width=True):
            st.session_state["selected_models"] = ["lda", "svm", "rf"]
            st.rerun()

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    selected_ids = []

    # ── 2-Column Playing Cards Grid ──
    col1, col2 = st.columns(2, gap="medium")
    model_items = list(registered.items())

    for idx, (model_id, model_def) in enumerate(model_items):
        target_col = col1 if idx % 2 == 0 else col2
        is_checked = model_id in st.session_state["selected_models"]
        configs = st.session_state["model_configs"].setdefault(model_id, {})
        info = MODEL_CARD_INFO.get(model_id, {
            "short_name": model_def.name,
            "full_name": model_def.name,
            "tag": "Classifier",
            "meta": "Standard ML",
            "desc": model_def.short_description
        })

        with target_col:
            with st.container(border=False, key=f"card_{model_id}"):
                # Row 1: Checkbox with name + tag beside it, and ⚙️ on the right
                c_chk, c_gear = st.columns([0.84, 0.16])
                with c_chk:
                    selected = st.checkbox(
                        f"**{info['short_name']}** `{info['tag']}`",
                        value=is_checked,
                        key=f"chk_{model_id}"
                    )
                    if selected:
                        selected_ids.append(model_id)
                with c_gear:
                    with st.popover("⚙️", use_container_width=True):
                        st.markdown(
                            f'<div style="font-size:13px;font-weight:700;letter-spacing:.04em;'
                            f'text-transform:uppercase;color:{name_color};margin:0 0 8px 0;">'
                            f'{info["short_name"]} Settings</div>',
                            unsafe_allow_html=True
                        )
                        _render_params(model_id, configs)

                # Row 2: Full model name (bigger text)
                row2_color = "#F8FAFC" if dark_mode else "#1E293B"
                st.markdown(
                    f'<div style="font-size:15px;font-weight:700;color:{row2_color};margin:4px 0 6px 0;">'
                    f'{info["full_name"]}</div>',
                    unsafe_allow_html=True
                )

                # Row 3: Description (bigger text)
                row3_color = "#CBD5E1" if dark_mode else "#475569"
                st.markdown(
                    f'<div style="font-size:13.5px;color:{row3_color};line-height:1.6;margin-bottom:10px;">'
                    f'{info["desc"]}</div>',
                    unsafe_allow_html=True
                )

                # Row 4: Bottom meta strip (bigger text & badges)
                if dark_mode:
                    strip_bg = "#1E293B"
                    strip_border = "#334155"
                    strip_color = "#94A3B8"
                    status_color = "#34D399" if is_checked else "#94A3B8"
                    status_bg = "rgba(16,185,129,0.25)" if is_checked else "#111827"
                    status_border = "rgba(52,211,153,0.3)" if is_checked else "#334155"
                else:
                    strip_bg = "#F8FAFC"
                    strip_border = "#E2E8F0"
                    strip_color = "#64748B"
                    status_color = "#047857" if is_checked else "#64748B"
                    status_bg = "#ECFDF5" if is_checked else "#F1F5F9"
                    status_border = "#A7F3D0" if is_checked else "#E2E8F0"

                status_text = "✓ Selected" if is_checked else "Available"
                st.markdown(
                    f'<div style="font-size:12.5px;font-weight:600;color:{strip_color};letter-spacing:.02em;'
                    f'background:{strip_bg};border:1px solid {strip_border};border-radius:8px;padding:7px 14px;'
                    f'margin-top:10px;display:flex;justify-content:space-between;align-items:center;">'
                    f'<span>{info["meta"]}</span>'
                    f'<span style="background:{status_bg};color:{status_color};border:1px solid {status_border};'
                    f'padding:3px 10px;border-radius:10px;font-size:12px;font-weight:700;">{status_text}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.session_state["selected_models"] = selected_ids

    # ── Action Bar (Compact right-aligned CTA) ──
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    col_back, col_spacer, col_train = st.columns([1, 2.3, 1.4])

    with col_back:
        if st.button("← Back to Dataset", use_container_width=True):
            st.session_state["current_step"] = 1
            st.rerun()

    with col_train:
        n = len(selected_ids)
        label = f"⚡ Train & Benchmark ({n}) →" if n else "Select at least 1"
        if st.button(label, type="primary", use_container_width=True, disabled=n == 0):
            st.session_state["is_training"] = True
            st.rerun()


def _render_training_screen(bundle, selected_ids: list, dark_mode: bool = False):
    """
    Dedicated Loading Screen:
    Renders ONLY the loading bar and verbose progress feedback underneath it.
    Zero modals, zero extra headers. Transitions directly to Step 3.
    """
    import time

    n_train = bundle.n_train_samples
    n_test = bundle.n_test_samples
    n_feat = bundle.n_features

    g_scaler = st.session_state.get("pipeline_params", {}).get("scaler", "standard")
    cv_k = st.session_state.get("pipeline_params", {}).get("cv_folds", 5)
    g_seed = st.session_state.get("pipeline_params", {}).get("random_seed", 42)
    g_weight = st.session_state.get("pipeline_params", {}).get("class_weight", None)

    scaler_display = (
        "StandardScaler" if g_scaler == "standard"
        else ("RobustScaler" if g_scaler == "robust"
              else ("MinMaxScaler" if g_scaler == "minmax" else "Raw (No Scaling)"))
    )

    stages = [
        {
            "done_text": f"Preprocessed feature matrix with {scaler_display} across {n_train} training trials",
            "active_text": f"Standardizing {n_train} EEG trials across {n_feat} spectral feature dimensions with {scaler_display}...",
        },
        {
            "done_text": f"Generated Stratified {cv_k}-Fold cross-validation splits with balanced class distributions (Seed: {g_seed})",
            "active_text": f"Partitioning Stratified {cv_k}-Fold cross-validation folds...",
        },
    ]

    for mid in selected_ids:
        info = MODEL_CARD_INFO.get(mid, {"short_name": mid.upper()})
        sname = info["short_name"]
        if mid == "lda":
            stages.append({
                "done_text": "Trained Linear Discriminant Analysis — estimated scatter matrices and Fisher eigenvalues",
                "active_text": "Training Linear Discriminant Analysis (LDA) — solving eigenvalue decomposition...",
            })
        elif mid == "svm":
            stages.append({
                "done_text": "Trained Support Vector Machine — RBF kernel Gram matrix evaluated, dual QP solver converged",
                "active_text": "Training Support Vector Machine (SVM) — solving dual quadratic program with RBF kernel...",
            })
        elif mid == "rf":
            stages.append({
                "done_text": "Trained Random Forest Classifier — 100 decorrelated decision trees bagged with Gini impurity splits",
                "active_text": "Training Random Forest (RF) — bagging 100 decorrelated trees across feature subspaces...",
            })
        elif mid == "knn":
            stages.append({
                "done_text": "Indexed K-Nearest Neighbors — built KD-Tree spatial index for Minkowski distance queries",
                "active_text": "Calibrating K-Nearest Neighbors (KNN) — indexing trial vectors in KD-Tree...",
            })
        elif mid == "lr":
            stages.append({
                "done_text": "Trained Logistic Regression — L-BFGS multinomial cross-entropy optimization converged",
                "active_text": "Fitting Logistic Regression (LR) — optimizing multinomial logit loss...",
            })
        else:
            stages.append({
                "done_text": f"Trained {sname} classifier and validated convergence",
                "active_text": f"Training {sname} classifier on feature matrix...",
            })

    stages.append({
        "done_text": f"Benchmarked test inference latency and jitter across {n_test} unseen EEG trials",
        "active_text": f"Running inference latency benchmarks across {n_test} test trials...",
    })
    stages.append({
        "done_text": "Computed evaluation metrics: Confusion Matrices, Cohen's Kappa, Balanced Accuracy & Macro-F1",
        "active_text": "Calculating Cohen's Kappa, Macro-F1, and per-class performance metrics...",
    })

    total_stages = len(stages)
    step_delay = 5.6 / float(total_stages)
    container = st.empty()
    exp = None

    # Theme palette
    card_bg = "#111827" if dark_mode else "#FFFFFF"
    card_border = "#334155" if dark_mode else "#CBD5E1"
    title_color = "#F8FAFC" if dark_mode else "#0F172A"
    pct_color = "#38BDF8" if dark_mode else "#2563EB"
    bar_track = "#1E293B" if dark_mode else "#EEF2F6"
    bar_fill = "#38BDF8" if dark_mode else "#2563EB"
    done_text_color = "#E2E8F0" if dark_mode else "#334155"
    active_bg = "rgba(37,99,235,0.25)" if dark_mode else "#EFF6FF"
    active_border = "rgba(147,197,253,0.3)" if dark_mode else "#BFDBFE"
    active_text_color = "#93C5FD" if dark_mode else "#1D4ED8"
    success_bg = "rgba(16,185,129,0.2)" if dark_mode else "#ECFDF5"
    success_border = "rgba(5,150,105,0.4)" if dark_mode else "#A7F3D0"
    success_text_color = "#6EE7B7" if dark_mode else "#065F46"
    card_shadow = (
        "0 4px 8px rgba(0,0,0,0.5), 0 12px 24px -2px rgba(0,0,0,0.6), 0 28px 48px -4px rgba(0,0,0,0.7)"
        if dark_mode else
        "0 1px 3px rgba(15,23,42,0.05), 0 4px 8px -1px rgba(15,23,42,0.08), 0 16px 32px -4px rgba(15,23,42,0.14), 0 24px 48px -8px rgba(15,23,42,0.08)"
    )

    for i in range(total_stages):
        pct = int(((i + 1) / total_stages) * 100)
        curr = stages[i]

        rows = []
        for j in range(i):
            rows.append(
                f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:13.5px;color:{done_text_color};line-height:1.5;">'
                f'<span style="color:#10B981;font-weight:800;font-size:15px;flex-shrink:0;margin-top:1px;">✓</span>'
                f'<span>{stages[j]["done_text"]}</span>'
                f'</div>'
            )

        rows.append(
            f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:13.5px;color:{active_text_color};font-weight:600;'
            f'line-height:1.5;background:{active_bg};border:1px solid {active_border};border-radius:8px;padding:9px 13px;">'
            f'<span class="term-cursor" style="flex-shrink:0;margin-top:1px;">●</span>'
            f'<span>{curr["active_text"]}</span>'
            f'</div>'
        )

        rows_html = "".join(rows)

        card_html = (
            f'<div style="background:{card_bg};border:1.5px solid {card_border};border-radius:14px;padding:28px 32px;'
            f'box-shadow:{card_shadow};'
            f'max-width:760px;margin:28px auto;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">'
            f'<span style="font-size:15px;font-weight:700;color:{title_color};">Training & Evaluating Classifiers...</span>'
            f'<span style="font-size:14px;font-weight:700;color:{pct_color};">{pct}%</span>'
            f'</div>'
            f'<div style="width:100%;height:8px;background:{bar_track};border-radius:999px;overflow:hidden;margin-bottom:22px;">'
            f'<div style="width:{pct}%;height:100%;background:{bar_fill};border-radius:999px;transition:width 0.3s ease;"></div>'
            f'</div>'
            f'<div style="display:flex;flex-direction:column;gap:10px;">'
            f'{rows_html}'
            f'</div>'
            f'</div>'
        )

        container.markdown(card_html, unsafe_allow_html=True)

        if i == total_stages - 2:
            try:
                active_configs = {}
                for mid in selected_ids:
                    cfg = dict(st.session_state.get("model_configs", {}).get(mid, {}))
                    cfg.setdefault("scaler_type", g_scaler)
                    cfg.setdefault("random_state", g_seed)
                    if g_weight is not None and "class_weight" not in cfg:
                        cfg["class_weight"] = g_weight
                    active_configs[mid] = cfg

                exp = train_and_evaluate(
                    bundle,
                    selected_ids,
                    active_configs
                )
            except Exception as e:
                st.session_state["is_training"] = False
                st.error(f"Training error: {e}")
                if st.button("← Return to Model Selection"):
                    st.rerun()
                return

        time.sleep(step_delay)

    all_done_rows = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:13.5px;color:{done_text_color};line-height:1.5;">'
        f'<span style="color:#10B981;font-weight:800;font-size:15px;flex-shrink:0;margin-top:1px;">✓</span>'
        f'<span>{s["done_text"]}</span>'
        f'</div>'
        for s in stages
    )

    final_card_html = (
        f'<div style="background:{card_bg};border:1.5px solid {card_border};border-radius:14px;padding:28px 32px;'
        f'box-shadow:{card_shadow};'
        f'max-width:760px;margin:28px auto;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">'
        f'<span style="font-size:15px;font-weight:700;color:{title_color};">Training & Evaluating Classifiers...</span>'
        f'<span style="font-size:14px;font-weight:700;color:#10B981;">100%</span>'
        f'</div>'
        f'<div style="width:100%;height:8px;background:{bar_track};border-radius:999px;overflow:hidden;margin-bottom:22px;">'
        f'<div style="width:100%;height:100%;background:#10B981;border-radius:999px;"></div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;gap:10px;">'
        f'{all_done_rows}'
        f'<div style="display:flex;align-items:center;gap:10px;font-size:13.5px;color:{success_text_color};font-weight:700;'
        f'background:{success_bg};border:1px solid {success_border};border-radius:8px;padding:10px 14px;margin-top:6px;">'
        f'<span style="font-size:15px;">✓</span>'
        f'<span>All classifiers successfully trained and benchmarked! Loading results...</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    container.markdown(final_card_html, unsafe_allow_html=True)
    time.sleep(0.5)

    if exp is not None:
        st.session_state["experiment_result"] = exp
        st.session_state["is_training"] = False
        st.session_state["current_step"] = 3
        st.rerun()


def _render_params(model_id: str, configs: dict):
    if model_id == "lda":
        col1, col2 = st.columns(2)
        with col1:
            solver = st.selectbox("Solver", ["svd", "lsqr", "eigen"], key=f"lda_solver_{model_id}",
                                  help="SVD is fastest; LSQR/EIGEN support shrinkage regularization")
            configs["solver"] = solver
        with col2:
            if solver != "svd":
                shrink = st.selectbox("Shrinkage", ["None", "auto (Ledoit-Wolf)"], key=f"lda_shrink_{model_id}")
                configs["shrinkage"] = None if shrink == "None" else "auto"
            else:
                st.caption("Shrinkage unavailable with SVD solver.")
                configs["shrinkage"] = None

    elif model_id == "svm":
        col1, col2, col3 = st.columns(3)
        with col1:
            configs["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly"], key=f"svm_k_{model_id}",
                                             help="RBF handles non-linear EEG boundaries; Linear is interpretable")
        with col2:
            configs["C"] = st.number_input("C (Regularization)", 0.01, 500.0,
                                           float(configs.get("C", 1.0)), 0.5, key=f"svm_c_{model_id}",
                                           help="Lower C = wider margin; higher C = strict boundary")
        with col3:
            configs["gamma"] = st.selectbox("Gamma", ["scale", "auto"], key=f"svm_g_{model_id}")
        st.caption("ℹ️  StandardScaler is applied automatically inside the SVM pipeline.")

    elif model_id == "rf":
        col1, col2, col3 = st.columns(3)
        with col1:
            configs["n_estimators"] = st.slider("Number of Trees", 20, 300, int(configs.get("n_estimators", 100)), 10,
                                                key=f"rf_n_{model_id}")
        with col2:
            d = st.selectbox("Max Tree Depth", ["None (Unlimited)", "3", "5", "8", "12"], key=f"rf_d_{model_id}")
            configs["max_depth"] = None if d.startswith("None") else int(d)
        with col3:
            cw = st.selectbox("Class Weight", ["None", "balanced"], key=f"rf_cw_{model_id}",
                              help="'balanced' adjusts for imbalanced EEG trial counts")
            configs["class_weight"] = None if cw == "None" else cw
        st.caption("ℹ️  Random Forest is scale-invariant; no feature normalization needed.")

    elif model_id == "knn":
        col1, col2, col3 = st.columns(3)
        with col1:
            configs["n_neighbors"] = st.slider("K (Neighbors)", 1, 25, int(configs.get("n_neighbors", 5)), 2,
                                               key=f"knn_k_{model_id}",
                                               help="Small K = complex boundary; large K = smoother")
        with col2:
            configs["metric"] = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"],
                                             key=f"knn_m_{model_id}")
        with col3:
            configs["weights"] = st.selectbox("Vote Weighting", ["uniform", "distance"], key=f"knn_w_{model_id}")
        st.caption("ℹ️  StandardScaler is applied automatically inside the KNN pipeline.")

    elif model_id == "lr":
        col1, col2, col3 = st.columns(3)
        with col1:
            configs["C"] = st.number_input("C (Inverse Regularization)", 0.01, 500.0,
                                           float(configs.get("C", 1.0)), 0.5, key=f"lr_c_{model_id}",
                                           help="Smaller C = stronger regularization (prevents overfitting)")
        with col2:
            configs["solver"] = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"], key=f"lr_s_{model_id}")
        with col3:
            configs["penalty"] = st.selectbox("Penalty", ["l2", "None"], key=f"lr_p_{model_id}")

    elif model_id == "riemann_mdm":
        configs["metric"] = st.selectbox("Riemannian Metric", ["riemann", "logeuclid", "euclid"],
                                         key=f"mdm_m_{model_id}",
                                         help="Only valid for covariance matrix input [N x Ch x Ch]")
        st.warning("MDM requires 3D covariance matrices as input, not flat feature vectors.")
