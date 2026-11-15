"""
Minimal layout shell. Header + stepper + router + dark mode toggle.
"""
import streamlit as st
from ui.styles import get_custom_css
from ui.components import render_header, render_stepper
from ui.data_page import render_data_page
from ui.model_page import render_model_page, _render_training_screen
from ui.results_page import render_results_page


def render_app_layout():
    is_training = st.session_state.get("is_training", False)

    # Injects theme CSS + instantaneously purges all stale cards and widgets if in training mode
    st.markdown(get_custom_css(dark_mode=False, is_training=is_training), unsafe_allow_html=True)

    bundle = st.session_state.get("dataset_bundle")

    # Render inside single slot container so transition completely wipes previous children
    page_slot = st.empty()
    with page_slot.container():
        # ── DEDICATED LOADING PAGE ──
        # When training/loading: ONLY render the breadcrumb (stepper) and the loading progress card.
        # No header, no modals, zero extra elements.
        if is_training:
            render_stepper(2)
            selected_ids = st.session_state.get("selected_models", ["lda", "svm", "rf"])
            _render_training_screen(bundle, selected_ids, dark_mode=False)
            return

        # ── NORMAL PAGE FLOW ──
        st.markdown(
            '<div style="font-size:42px;font-weight:900;letter-spacing:-0.035em;'
            'color:#0F172A;text-align:center;margin:0 0 16px 0;line-height:1.1;'
            'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,\'Inter\',sans-serif;">'
            'Classification Module</div>',
            unsafe_allow_html=True
        )

        if "current_step" not in st.session_state:
            st.session_state["current_step"] = 1

        render_stepper(st.session_state["current_step"])

        step = st.session_state["current_step"]
        if step == 1:
            render_data_page()
        elif step == 2:
            render_model_page()
        elif step == 3:
            render_results_page()

        # Clean unobtrusive footer with one-line run command
        st.markdown(
            '<div style="text-align:center;margin-top:40px;padding-top:16px;border-top:1px solid #E2E8F0;'
            'font-size:12px;color:#94A3B8;letter-spacing:0.02em;">'
            'Module 8: EEG Classification Lab &nbsp;•&nbsp; '
            'One-line run command: <code style="background:#F1F5F9;color:#334155;padding:2px 6px;border-radius:4px;font-size:11.5px;">streamlit run app.py</code>'
            '</div>',
            unsafe_allow_html=True
        )
