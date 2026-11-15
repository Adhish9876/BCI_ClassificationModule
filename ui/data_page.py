"""
Data page — Unicorn UX: clean visual hierarchy, uncluttered tabs, guided flow.
"""
import pandas as pd
import streamlit as st
from classification.data_loader import load_csv_data, prepare_dataset_from_separate_dfs, prepare_dataset_from_single_df
from classification.sample_data import get_default_sample_bundle
from classification.validation import validate_dataset
from ui.components import plot_class_distribution, plot_feature_distribution


def _card_html(title, desc, class_badges_html, stat_badges_html, is_active):
    """Build dataset card HTML with depth, layered shadows, and pop."""
    dark_mode = st.session_state.get("dark_mode", False)
    if dark_mode:
        border = "2px solid #3B82F6" if is_active else "1px solid #334155"
        bg = "#111827"
        title_color = "#F8FAFC"
        desc_color = "#CBD5E1"
        shadow = (
            "0 2px 8px rgba(59,130,246,0.25), 0 10px 24px -2px rgba(59,130,246,0.35), 0 24px 44px -4px rgba(59,130,246,0.30)"
            if is_active
            else "0 2px 4px rgba(0,0,0,0.4), 0 8px 16px -2px rgba(0,0,0,0.5), 0 20px 36px -4px rgba(0,0,0,0.6)"
        )
    else:
        border = "2px solid #10B981" if is_active else "1.5px solid #CBD5E1"
        bg = "#FFFFFF"
        title_color = "#0F172A"
        desc_color = "#475569"
        shadow = (
            "0 1px 3px rgba(15,23,42,0.06), 0 4px 12px -2px rgba(15,23,42,0.08)"
            if is_active
            else "0 1px 3px rgba(15,23,42,0.05), 0 4px 8px -1px rgba(15,23,42,0.08), 0 12px 24px -4px rgba(15,23,42,0.12), 0 20px 32px -6px rgba(15,23,42,0.06)"
        )
    transform = "transform:translateY(-2px);" if is_active else ""
    active_pill = ""
    if is_active:
        active_pill = (
            '<span style="display:inline-flex;align-items:center;gap:6px;padding:3px 10px;'
            'border-radius:12px;font-size:11px;font-weight:700;letter-spacing:.04em;'
            'text-transform:uppercase;background:rgba(37,99,235,0.25);color:#93C5FD;border:1px solid rgba(147,197,253,0.3);'
            'box-shadow:0 1px 3px rgba(37,99,235,0.15);">'
            '<span style="width:6px;height:6px;border-radius:50%;background:#38BDF8;display:inline-block;"></span>'
            'Active</span>'
        )
    return (
        f'<div style="background:{bg};border-radius:14px;padding:22px 24px;border:{border};'
        f'box-shadow:{shadow};{transform}transition:all .2s cubic-bezier(0.16,1,0.3,1);height:100%;">'
        f'<div style="font-size:16px;font-weight:700;color:{title_color};margin-bottom:8px;'
        f'display:flex;justify-content:space-between;align-items:center;">'
        f'<span>{title}</span>{active_pill}</div>'
        f'<div style="font-size:13px;color:{desc_color};line-height:1.5;margin-bottom:14px;">{desc}</div>'
        f'<div style="margin-bottom:12px;">{class_badges_html}</div>'
        f'<div>{stat_badges_html}</div>'
        f'</div>'
    )


def _badge(text, variant="slate"):
    """Render a small inline badge."""
    colors = {
        "blue":    ("background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;",),
        "indigo":  ("background:#EEF2FF;color:#4338CA;border:1px solid #C7D2FE;",),
        "amber":   ("background:#FFFBEB;color:#B45309;border:1px solid #FDE68A;",),
        "emerald": ("background:#ECFDF5;color:#047857;border:1px solid #A7F3D0;",),
        "slate":   ("background:#F8FAFC;color:#475569;border:1px solid #E2E8F0;",),
    }
    s = colors.get(variant, colors["slate"])[0]
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:11px;font-weight:600;margin:2px 4px 2px 0;{s}">{text}</span>'
    )


def _render_dark_preview_table(df: pd.DataFrame) -> str:
    """Render feature preview table in dark terminal matrix mode."""
    cols = list(df.columns)
    th_cells = [
        '<th style="padding:10px 14px;color:#64748B;font-weight:700;text-align:center;'
        'border-bottom:1px solid #1E293B;border-right:1px solid #1E293B;position:sticky;'
        'top:0;background:#111827;z-index:10;font-size:11px;">#</th>'
    ]
    for c in cols:
        align = "left" if c in ("Class Label", "Participant") else "right"
        th_cells.append(
            f'<th style="padding:10px 14px;color:#94A3B8;font-weight:700;text-align:{align};'
            f'border-bottom:1px solid #1E293B;position:sticky;top:0;background:#111827;z-index:10;'
            f'font-size:11px;letter-spacing:.03em;font-family:sans-serif;">{c}</th>'
        )
    th_html = "".join(th_cells)

    tr_rows = []
    for idx, row in df.iterrows():
        bg = "#0B0F19" if idx % 2 == 0 else "#0F172A"
        td_cells = [
            f'<td style="padding:8px 14px;color:#475569;font-weight:600;text-align:center;'
            f'border-right:1px solid #1E293B;border-bottom:1px solid #1E293B;background:{bg};'
            f'font-size:11px;">{idx}</td>'
        ]
        for c in cols:
            val = row[c]
            if c == "Class Label":
                val_str = str(val)
                if "Rest" in val_str:
                    color = "#34D399"
                elif "Left" in val_str:
                    color = "#FBBF24"
                else:
                    color = "#60A5FA"
                td_cells.append(
                    f'<td style="padding:8px 14px;font-weight:600;color:{color};text-align:left;'
                    f'border-bottom:1px solid #1E293B;background:{bg};font-size:12px;">{val_str}</td>'
                )
            elif c == "Participant":
                td_cells.append(
                    f'<td style="padding:8px 14px;color:#94A3B8;text-align:left;'
                    f'border-bottom:1px solid #1E293B;background:{bg};font-size:12px;">{val}</td>'
                )
            else:
                try:
                    num_str = f"{float(val):.6f}"
                except (ValueError, TypeError):
                    num_str = str(val)
                td_cells.append(
                    f'<td style="padding:8px 14px;color:#F1F5F9;text-align:right;'
                    f'border-bottom:1px solid #1E293B;background:{bg};font-family:Consolas, Monaco, monospace;'
                    f'font-size:12px;">{num_str}</td>'
                )
        tr_rows.append(f'<tr>{"".join(td_cells)}</tr>')

    tbody_html = "".join(tr_rows)

    return (
        f'<div class="dark-preview-table-wrap" style="background:#0B0F19;border:1px solid #1E293B;'
        f'border-radius:8px;overflow-x:auto;overflow-y:auto;max-height:300px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.25);">'
        f'<table style="width:100%;border-collapse:collapse;white-space:nowrap;">'
        f'<thead><tr>{th_html}</tr></thead>'
        f'<tbody>{tbody_html}</tbody>'
        f'</table>'
        f'</div>'
    )


def render_data_page():
    # ── Init session state ──
    if "dataset_loaded" not in st.session_state:
        st.session_state["dataset_loaded"] = False
    if "selected_dataset_type" not in st.session_state:
        st.session_state["selected_dataset_type"] = None

    # ── Section label ──
    st.markdown(
        '<div style="font-size:22px;font-weight:800;letter-spacing:-0.02em;'
        'color:#0F172A;margin:24px 0 16px 0;padding-bottom:8px;border-bottom:2px solid #E2E8F0;'
        'display:flex;align-items:center;gap:10px;">'
        '<span>📂</span><span>Dataset Selection</span></div>',
        unsafe_allow_html=True
    )

    # ── Source selector — TABS ──
    tab_sample, tab_upload = st.tabs(["✦  Sample Datasets", "📁  Upload Your Own"])

    with tab_sample:
        st.markdown("""
        <style>
        /* Scoped green border for Load Dataset buttons - text remains dark initially */
        div[class*="st-key-btn2"] button,
        div[class*="st-key-btn3"] button,
        div[class*="st-key-sample_card_"] button,
        div[class*="st-key-sample_card_"] .stButton > button {
            border: 2px solid #10B981 !important;
            border-color: #10B981 !important;
            color: #0F172A !important;
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 13.5px !important;
            border-radius: 8px !important;
            padding: 7px 16px !important;
            box-shadow: 0 1px 3px rgba(16, 185, 129, 0.15) !important;
            transition: all 0.2s ease !important;
        }
        div[class*="st-key-btn2"] button:hover,
        div[class*="st-key-btn3"] button:hover,
        div[class*="st-key-sample_card_"] button:hover,
        div[class*="st-key-sample_card_"] .stButton > button:hover {
            border: 2px solid #059669 !important;
            border-color: #059669 !important;
            background: #F0FDF4 !important;
            background-color: #F0FDF4 !important;
            color: #0F172A !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
            transform: translateY(-1px) !important;
        }
        div[class*="st-key-btn2"] button p,
        div[class*="st-key-btn3"] button p,
        div[class*="st-key-sample_card_"] button p {
            color: #0F172A !important;
            font-weight: 700 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")

        is_2class_active = st.session_state.get("selected_dataset_type") == "2class"
        is_3class_active = st.session_state.get("selected_dataset_type") == "3class"

        with col1:
            card_key = "sample_card_2class_active" if is_2class_active else "sample_card_2class"
            with st.container(border=False, key=card_key):
                active_pill = (
                    '<span style="background:#ECFDF5;color:#059669;border:1px solid #A7F3D0;'
                    'padding:3px 10px;border-radius:12px;font-size:11.5px;font-weight:700;">'
                    '● Active</span>'
                ) if is_2class_active else ""
                st.markdown(
                    f'<div style="font-size:18px;font-weight:800;color:#0F172A;margin-bottom:6px;'
                    f'display:flex;justify-content:space-between;align-items:center;">'
                    f'<span>2-Class Motor Imagery</span>{active_pill}</div>'
                    f'<div style="font-size:13.5px;color:#475569;line-height:1.5;margin-bottom:12px;">'
                    f'Binary classification: brain at rest versus active motor imagery.</div>'
                    f'<div style="margin-bottom:16px;">'
                    f'{_badge("Rest", "emerald")}{_badge("Motor Imagery", "blue")}</div>',
                    unsafe_allow_html=True
                )
                c_stat, c_btn = st.columns([1.3, 1.2])
                with c_stat:
                    st.markdown(
                        f'<div style="padding-top:6px;">'
                        f'{_badge("800 trials")}{_badge("16 features")}{_badge("8 subjects")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with c_btn:
                    if is_2class_active:
                        st.markdown(
                            '<div style="background:#10B981;border:2px solid #059669;color:#FFFFFF;'
                            'border-radius:8px;padding:7px 14px;font-weight:700;font-size:13.5px;'
                            'display:flex;align-items:center;justify-content:center;gap:6px;text-align:center;'
                            'box-shadow:0 2px 8px rgba(16,185,129,0.35);">'
                            '<span style="color:#FFFFFF;font-weight:700;">✓ Loaded</span></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        if st.button("Load Dataset", key="btn2", use_container_width=True):
                            st.session_state["selected_dataset_type"] = "2class"
                            st.session_state["dataset_bundle"] = get_default_sample_bundle(n_samples=800, n_features=16, n_classes=2)
                            st.session_state["dataset_loaded"] = True
                            st.rerun()

        with col2:
            card_key = "sample_card_3class_active" if is_3class_active else "sample_card_3class"
            with st.container(border=False, key=card_key):
                active_pill = (
                    '<span style="background:#ECFDF5;color:#059669;border:1px solid #A7F3D0;'
                    'padding:3px 10px;border-radius:12px;font-size:11.5px;font-weight:700;">'
                    '● Active</span>'
                ) if is_3class_active else ""
                st.markdown(
                    f'<div style="font-size:18px;font-weight:800;color:#0F172A;margin-bottom:6px;'
                    f'display:flex;justify-content:space-between;align-items:center;">'
                    f'<span>3-Class Motor Imagery</span>{active_pill}</div>'
                    f'<div style="font-size:13.5px;color:#475569;line-height:1.5;margin-bottom:12px;">'
                    f'3-way classification: rest, left-hand imagery, right-hand imagery.</div>'
                    f'<div style="margin-bottom:16px;">'
                    f'{_badge("Rest", "emerald")}{_badge("Left Hand", "amber")}{_badge("Right Hand", "blue")}</div>',
                    unsafe_allow_html=True
                )
                c_stat, c_btn = st.columns([1.3, 1.2])
                with c_stat:
                    st.markdown(
                        f'<div style="padding-top:6px;">'
                        f'{_badge("900 trials")}{_badge("18 features")}{_badge("9 subjects")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with c_btn:
                    if is_3class_active:
                        st.markdown(
                            '<div style="background:#10B981;border:2px solid #059669;color:#FFFFFF;'
                            'border-radius:8px;padding:7px 14px;font-weight:700;font-size:13.5px;'
                            'display:flex;align-items:center;justify-content:center;gap:6px;text-align:center;'
                            'box-shadow:0 2px 8px rgba(16,185,129,0.35);">'
                            '<span style="color:#FFFFFF;font-weight:700;">✓ Loaded</span></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        if st.button("Load Dataset", key="btn3", use_container_width=True):
                            st.session_state["selected_dataset_type"] = "3class"
                            st.session_state["dataset_bundle"] = get_default_sample_bundle(n_samples=900, n_features=18, n_classes=3)
                            st.session_state["dataset_loaded"] = True
                            st.rerun()

    with tab_upload:
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        upload_mode = st.selectbox(
            "Upload format",
            ["Single CSV file (with class label column)", "Four separate CSVs (X_train, y_train, X_test, y_test)"],
            key="upload_mode_sel"
        )

        if "Single" in upload_mode:
            f = st.file_uploader("Upload feature matrix CSV", type=["csv"], key="single_up")
            if f:
                df = load_csv_data(f)
                st.dataframe(df.head(3), use_container_width=True)

                col_a, col_b, col_c = st.columns([2, 2, 1])
                with col_a:
                    target = st.selectbox("Class label column", list(df.columns), index=len(df.columns)-1)
                with col_b:
                    grp_opts = ["None (standard stratified split)"] + [c for c in df.columns if c != target]
                    grp = st.selectbox("Subject / Participant ID column (optional)", grp_opts)
                with col_c:
                    split = st.number_input("Test Split %", min_value=10, max_value=40, value=20, step=5)

                if st.button("Validate & Load Custom Dataset", type="primary"):
                    try:
                        b = prepare_dataset_from_single_df(
                            df, target, test_size=split/100,
                            group_column=None if grp.startswith("None") else grp,
                            dataset_name=f.name
                        )
                        st.session_state["dataset_bundle"] = b
                        st.session_state["dataset_loaded"] = True
                        st.session_state["selected_dataset_type"] = "upload"
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                fx_tr = st.file_uploader("X_train.csv", type=["csv"])
                fy_tr = st.file_uploader("y_train.csv", type=["csv"])
            with col_u2:
                fx_te = st.file_uploader("X_test.csv", type=["csv"])
                fy_te = st.file_uploader("y_test.csv", type=["csv"])
            if all([fx_tr, fy_tr, fx_te, fy_te]):
                if st.button("Validate & Load Custom Dataset", type="primary"):
                    try:
                        b = prepare_dataset_from_separate_dfs(fx_tr, fy_tr, fx_te, fy_te)
                        st.session_state["dataset_bundle"] = b
                        st.session_state["dataset_loaded"] = True
                        st.session_state["selected_dataset_type"] = "upload"
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    # ── Guidance when no dataset is loaded ──
    if not st.session_state.get("dataset_loaded", False):
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;'
            'padding:16px 20px;font-size:14px;font-weight:600;color:#1E40AF;'
            'display:flex;align-items:center;gap:10px;">'
            '<span>👆</span>'
            '<span>Pick a sample dataset above and click Load Dataset to get started.</span>'
            '</div>',
            unsafe_allow_html=True
        )
        return

    # ── Dataset Overview ──
    bundle = st.session_state["dataset_bundle"]
    dark_mode = st.session_state.get("dark_mode", False)

    section_header_color = "#F8FAFC" if dark_mode else "#334155"
    border_color = "#334155" if dark_mode else "#E2E8F0"
    stat_bg = "#111827" if dark_mode else "#FFFFFF"
    stat_num_color = "#F8FAFC" if dark_mode else "#0F172A"
    stat_label_color = "#94A3B8" if dark_mode else "#64748B"
    stat_border = "#334155" if dark_mode else "#E2E8F0"

    st.markdown(f"<hr style='border:none;border-top:1px solid {border_color};margin:28px 0;'>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:22px;font-weight:800;letter-spacing:-0.02em;'
        f'color:{section_header_color};margin:28px 0 16px 0;padding-bottom:8px;border-bottom:2px solid {border_color};'
        f'display:flex;align-items:center;gap:10px;">'
        f'<span>📊</span><span>Dataset Overview</span></div>',
        unsafe_allow_html=True
    )

    report = validate_dataset(bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test)
    if not report.is_valid:
        for e in report.errors:
            st.error(e)

    # ── Stat strip (depth & elevation) ──
    class_badges = "".join(_badge(c, "blue") for c in bundle.class_names)
    stat_shadow = (
        "0 2px 4px rgba(0,0,0,0.4), 0 8px 16px -2px rgba(0,0,0,0.5), 0 20px 36px -4px rgba(0,0,0,0.6)"
        if dark_mode else
        "0 1px 3px rgba(15,23,42,0.05), 0 4px 8px -1px rgba(15,23,42,0.08), 0 12px 24px -4px rgba(15,23,42,0.12), 0 20px 32px -6px rgba(15,23,42,0.06)"
    )
    stat_html = (
        f'<div style="background:{stat_bg};border-radius:14px;border:1.5px solid {stat_border};'
        f'padding:20px 28px;margin:18px 0;display:flex;gap:28px;align-items:center;'
        f'box-shadow:{stat_shadow};">'
        f'<div style="text-align:left;">'
        f'<span style="font-size:24px;font-weight:800;color:{stat_num_color};display:block;letter-spacing:-0.02em;line-height:1.2;">{bundle.n_train_samples}</span>'
        f'<span style="font-size:11px;font-weight:600;color:{stat_label_color};text-transform:uppercase;letter-spacing:.05em;margin-top:3px;">Training Trials</span>'
        f'</div>'
        f'<div style="width:1px;height:36px;background:{stat_border};flex-shrink:0;"></div>'
        f'<div style="text-align:left;">'
        f'<span style="font-size:24px;font-weight:800;color:{stat_num_color};display:block;letter-spacing:-0.02em;line-height:1.2;">{bundle.n_test_samples}</span>'
        f'<span style="font-size:11px;font-weight:600;color:{stat_label_color};text-transform:uppercase;letter-spacing:.05em;margin-top:3px;">Test Trials</span>'
        f'</div>'
        f'<div style="width:1px;height:36px;background:{stat_border};flex-shrink:0;"></div>'
        f'<div style="text-align:left;">'
        f'<span style="font-size:24px;font-weight:800;color:{stat_num_color};display:block;letter-spacing:-0.02em;line-height:1.2;">{bundle.n_features}</span>'
        f'<span style="font-size:11px;font-weight:600;color:{stat_label_color};text-transform:uppercase;letter-spacing:.05em;margin-top:3px;">EEG Features</span>'
        f'</div>'
        f'<div style="width:1px;height:36px;background:{stat_border};flex-shrink:0;"></div>'
        f'<div style="text-align:left;">'
        f'<span style="font-size:24px;font-weight:800;color:{stat_num_color};display:block;letter-spacing:-0.02em;line-height:1.2;">{bundle.n_classes}</span>'
        f'<span style="font-size:11px;font-weight:600;color:{stat_label_color};text-transform:uppercase;letter-spacing:.05em;margin-top:3px;">Classes</span>'
        f'</div>'
        f'<div style="width:1px;height:36px;background:{stat_border};flex-shrink:0;"></div>'
        f'<div style="flex:1;">'
        f'<div>{class_badges}</div>'
        f'<span style="font-size:11px;font-weight:600;color:{stat_label_color};text-transform:uppercase;letter-spacing:.05em;display:block;margin-top:4px;">Class Labels</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(stat_html, unsafe_allow_html=True)

    # ── Linked Feature Exploration Section Heading ──
    st.markdown(
        f'<div style="font-size:22px;font-weight:800;letter-spacing:-0.02em;'
        f'color:#0F172A;margin:36px 0 16px 0;padding-bottom:8px;border-bottom:2px solid {border_color};'
        f'display:flex;align-items:center;justify-content:space-between;">'
        f'<span>Feature Inspection & Linked Distributions</span>'
        f'<span style="background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;font-size:12px;font-weight:700;padding:3px 10px;border-radius:6px;">'
        f'Live Synchronized</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.4, 1.1, 1.0])
    with col_ctrl1:
        chosen_feat = st.selectbox(
            "Select EEG Feature to Live-Inspect:",
            bundle.feature_names,
            key="linked_feat_sel",
            help="Selecting any feature immediately recomputes its class separation and distributions across partitions"
        )
    with col_ctrl2:
        split_opt = st.radio(
            "Data Partition View:",
            ["All Trials", "Train Split Only", "Test Split Only"],
            horizontal=True,
            key="linked_split_opt"
        )
    with col_ctrl3:
        p2_view = st.selectbox(
            "Plot 2 Representation:",
            ["Histogram (Distribution)", "Box Plot (Quartiles)", "Violin Plot (Density)"],
            index=0,
            key="linked_p2_view",
            help="Switch visualization style to view probability density, statistical quartiles, or kernel shape"
        )

    # Filter data based on partition
    if split_opt == "Train Split Only":
        X_sub, y_sub = bundle.X_train, bundle.y_train
    elif split_opt == "Test Split Only":
        X_sub, y_sub = bundle.X_test, bundle.y_test
    else:
        X_sub = pd.concat([bundle.X_train, bundle.X_test], ignore_index=True)
        y_sub = pd.concat([bundle.y_train, bundle.y_test], ignore_index=True)

    col_plot1, col_plot2 = st.columns([1, 1.2], gap="medium")
    with col_plot1:
        st.markdown(f'<div style="font-size:16px;font-weight:800;color:#0F172A;margin-bottom:8px;">Plot 1: Class Balance ({split_opt})</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_class_distribution(bundle.y_train, bundle.y_test),
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
            key="chart_data_class_distribution"
        )
    with col_plot2:
        st.markdown(f'<div style="font-size:16px;font-weight:800;color:#0F172A;margin-bottom:8px;">Plot 2: {chosen_feat} Separation by Class</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_feature_distribution(X_sub, y_sub, chosen_feat, chart_type=p2_view),
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
            key=f"chart_data_feat_dist_{chosen_feat}_{p2_view}"
        )

    # ── Sample Preview Table ──
    st.markdown(
        f'<div style="font-size:22px;font-weight:800;letter-spacing:-0.02em;'
        f'color:#0F172A;margin:38px 0 16px 0;padding-bottom:8px;border-bottom:2px solid {border_color};">'
        f'<span>Sample Preview — Features & Labels (First 8 Rows)</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    preview_df = bundle.X_train.copy().reset_index(drop=True)
    label_series = bundle.y_train.reset_index(drop=True)
    preview_df.insert(0, "Class Label", label_series)
    if hasattr(bundle, "groups_train") and bundle.groups_train is not None:
        preview_df.insert(1, "Participant", bundle.groups_train.reset_index(drop=True))
    st.markdown(_render_dark_preview_table(preview_df.head(8)), unsafe_allow_html=True)

    # ── Bottom Action Bar (Clean CTA, no redundant active dataset strip) ──
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    col_l, col_btn = st.columns([2.4, 1.3])
    with col_btn:
        if st.button("Select Classifiers →", type="primary", use_container_width=True):
            st.session_state["current_step"] = 2
            st.rerun()
