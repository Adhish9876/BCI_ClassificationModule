"""
Export and serialization utilities for the EEG Classification Lab.
Supports:
1. Model comparison results as CSV.
2. Complete experiment configuration and metrics as JSON.
3. Detailed per-model classification reports.
4. Serialized trained pipelines via joblib with security warnings.
"""

import io
import json
from typing import Any, Dict, Optional
import joblib
import pandas as pd

from .schemas import EvaluationResult, ExperimentResult


def export_comparison_csv(experiment_result: ExperimentResult) -> str:
    """Exports model comparison table as CSV text."""
    df = experiment_result.to_dataframe()
    return df.to_csv(index=False)


def export_experiment_json(experiment_result: ExperimentResult, indent: int = 2) -> str:
    """Exports full experiment metadata, configurations, and results as JSON text."""
    data = experiment_result.to_dict()
    return json.dumps(data, indent=indent, default=str)


def export_classification_report_csv(result: EvaluationResult) -> str:
    """Exports per-class metrics (precision, recall, f1, support) as CSV text."""
    if not result.per_class_metrics:
        return "class,precision,recall,f1_score,support\n"

    rows = []
    for cls_name, metrics in result.per_class_metrics.items():
        row = {"Class": cls_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def serialize_trained_model(result: EvaluationResult) -> bytes:
    """
    Serializes the scikit-learn Pipeline using joblib into in-memory bytes.
    
    WARNING:
    Pickled/joblib model files execute arbitrary code upon deserialization.
    Only deserialize models obtained from trusted, verified sources.
    """
    if result.trained_pipeline is None:
        raise ValueError(f"No trained pipeline found for model {result.model_name}")

    buffer = io.BytesIO()
    joblib.dump(result.trained_pipeline, buffer)
    return buffer.getvalue()


def export_experiment_parameters_json(experiment_result: ExperimentResult, indent: int = 2) -> str:
    """
    Exports a clean, structured JSON configuration containing all adjustable parameters,
    hyperparameter dictionaries, dataset properties, and execution parameters.
    """
    params_payload = {
        "experiment_name": f"BCI_Classification_{experiment_result.dataset_name}",
        "timestamp": experiment_result.timestamp,
        "dataset_metadata": {
            "dataset_name": experiment_result.dataset_name,
            "source_type": experiment_result.source_type,
            "n_train_samples": experiment_result.n_train_samples,
            "n_test_samples": experiment_result.n_test_samples,
            "n_features": experiment_result.n_features,
            "classes": experiment_result.classes,
            "feature_names": experiment_result.feature_names
        },
        "model_hyperparameters": {
            r.model_id: {
                "model_name": r.model_name,
                "hyperparameters": r.hyperparameters,
                "is_success": r.is_success
            }
            for r in experiment_result.model_results.values()
        },
        "system_metadata": experiment_result.metadata
    }
    return json.dumps(params_payload, indent=indent, default=str)


def export_reproducible_python_code(
    experiment_result: ExperimentResult,
    dataset_name: Optional[str] = None
) -> str:
    """
    Generates a standalone, self-contained executable Python script (.py) that
    builds the scikit-learn pipelines with the exact configured hyperparameters,
    trains them, and computes identical evaluation benchmarks outside Streamlit.
    """
    d_name = dataset_name or experiment_result.dataset_name
    n_feats = experiment_result.n_features
    classes = experiment_result.classes

    # Build model configurations dictionary string
    configs = {}
    for r in experiment_result.model_results.values():
        if r.is_success:
            configs[r.model_id] = r.hyperparameters

    code_template = f'''"""
================================================================================
EEG Classification Lab — Reproducible Training Script
Generated: {experiment_result.timestamp}
Dataset: {d_name} ({n_feats} EEG features, Classes: {classes})
Total Pipeline Duration: {experiment_result.total_duration_sec:.2f} s
================================================================================
This script reproduces the exact scikit-learn training and evaluation pipeline
executed in the BCI Classification Lab application.

Requirements:
    pip install numpy pandas scikit-learn
================================================================================
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix
)

# ------------------------------------------------------------------------------
# 1. Configured Hyperparameters
# ------------------------------------------------------------------------------
HYPERPARAMETERS = {json.dumps(configs, indent=4)}


# ------------------------------------------------------------------------------
# 2. Pipeline Constructors
# ------------------------------------------------------------------------------
def create_model_pipeline(model_id: str, params: dict) -> Pipeline:
    """Builds the leakage-free scikit-learn Pipeline for a classifier."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
    
    if model_id == "lda":
        solver = params.get("solver", "svd")
        shrinkage = params.get("shrinkage", None)
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    elif model_id == "svm":
        clf = SVC(
            kernel=params.get("kernel", "rbf"),
            C=float(params.get("C", 1.0)),
            gamma=params.get("gamma", "scale"),
            degree=int(params.get("degree", 3)),
            probability=True,
            random_state=42
        )
    elif model_id == "rf":
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth", None),
            class_weight=params.get("class_weight", None),
            random_state=42
        )
    elif model_id == "knn":
        clf = KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 5)),
            metric=params.get("metric", "euclidean"),
            weights=params.get("weights", "uniform")
        )
    elif model_id == "lr":
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver=params.get("solver", "lbfgs"),
            penalty=params.get("penalty", "l2"),
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model identifier: {{model_id}}")
        
    steps.append(("classifier", clf))
    return Pipeline(steps)


# ------------------------------------------------------------------------------
# 3. Main Training & Benchmarking Routine
# ------------------------------------------------------------------------------
def run_experiment(X_train, y_train, X_test, y_test):
    print("=" * 70)
    print("EEG Classification Lab — Benchmark Execution")
    print(f"Training Samples: {{len(X_train)}}, Test Samples: {{len(X_test)}}")
    print(f"Number of Features: {{X_train.shape[1]}}")
    print("=" * 70)

    results_table = []

    for model_id, params in HYPERPARAMETERS.items():
        print(f"\\nFitting {{model_id.upper()}} pipeline...")
        pipe = create_model_pipeline(model_id, params)

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        pred_time = time.perf_counter() - t1

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)

        results_table.append({{
            "Model": model_id.upper(),
            "Accuracy": f"{{acc:.4f}}",
            "Balanced Acc": f"{{bal_acc:.4f}}",
            "Macro F1": f"{{f1_macro:.4f}}",
            "Cohen Kappa": f"{{kappa:.4f}}",
            "Fit Time (s)": f"{{fit_time:.3f}}",
            "Inference Time (ms)": f"{{pred_time * 1000:.2f}}"
        }})

        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for {{model_id.upper()}}:\\n{{cm}}")

    print("\\n" + "=" * 70)
    print("FINAL COMPARISON BENCHMARK:")
    print("=" * 70)
    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    print("Please load or provide your X_train, y_train, X_test, y_test data.")
    print("Example: X_train = pd.read_csv('X_train.csv')")
'''
    return code_template
