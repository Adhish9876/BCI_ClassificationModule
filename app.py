"""
EEG Classification Lab - Main Application Entrypoint
Provides both standalone execution and embeddable module integration for the BCI teaching suite.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import streamlit as st

from classification.data_loader import prepare_dataset_from_arrays
from classification.sample_data import get_default_sample_bundle
from classification.schemas import DatasetBundle
from ui.layout import render_app_layout


def render_classification_module(
    dataset: Optional[Union[DatasetBundle, Dict[str, Any]]] = None
):
    """
    Public integration interface for the Classification Module.
    
    Other student modules (specifically Feature Extraction) can embed this module 
    by invoking:
        render_classification_module(dataset={
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': [...],   # Optional
            'class_names': [...]      # Optional
        })
    
    If `dataset` is None, the module operates independently using its built-in
    synthetic generator or custom file upload facilities.
    """
    # Initialize session state bundle if incoming dataset is provided
    if dataset is not None:
        if isinstance(dataset, DatasetBundle):
            st.session_state["dataset_bundle"] = dataset
            st.session_state["integrated_dataset"] = dataset
        elif isinstance(dataset, dict):
            # Parse dictionary contract
            if all(k in dataset for k in ("X_train", "y_train", "X_test", "y_test")):
                bundle = prepare_dataset_from_arrays(
                    X_train=dataset["X_train"],
                    y_train=dataset["y_train"],
                    X_test=dataset["X_test"],
                    y_test=dataset["y_test"],
                    feature_names=dataset.get("feature_names"),
                    class_names=dataset.get("class_names"),
                    dataset_name=dataset.get("dataset_name", "Upstream Feature Extraction Data")
                )
                st.session_state["dataset_bundle"] = bundle
                st.session_state["integrated_dataset"] = bundle

    # Fallback to default sample dataset if none exists in session
    if "dataset_bundle" not in st.session_state or st.session_state["dataset_bundle"] is None:
        st.session_state["dataset_bundle"] = get_default_sample_bundle()

    # Render main UI layout
    render_app_layout()


def main():
    """Configures page options and runs the standalone Streamlit application."""
    st.set_page_config(
        page_title="EEG Classification Lab",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    render_classification_module()


if __name__ == "__main__":
    main()
