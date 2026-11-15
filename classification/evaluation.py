"""
Model evaluation module for EEG Classification Lab.
Computes comprehensive metrics using scikit-learn:
Accuracy, Balanced Accuracy, Macro/Weighted Precision, Recall, F1, Cohen's Kappa,
Confusion Matrix, per-class metrics, ROC-AUC, prediction time, and feature importances.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline

from .schemas import EvaluationResult


def evaluate_classifier(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_id: str,
    model_name: str,
    hyperparameters: Dict[str, Any],
    train_time_sec: float = 0.0,
    class_order: Optional[List[str]] = None
) -> EvaluationResult:
    """
    Evaluates a trained classifier pipeline against test data.
    Computes all standard BCI comparative performance metrics without approximations.
    """
    start_pred = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time_sec = time.perf_counter() - start_pred

    # Ensure y_test and y_pred have consistent types (convert to string or int representation)
    y_true_clean = y_test.values
    
    # Unique classes
    if class_order is not None:
        classes = [str(c) for c in class_order]
    else:
        classes = sorted(list({str(c) for c in np.unique(y_true_clean)} | {str(c) for c in np.unique(y_pred)}))

    # Standard metrics
    acc = float(accuracy_score(y_true_clean, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true_clean, y_pred))

    prec_macro = float(precision_score(y_true_clean, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true_clean, y_pred, average="weighted", zero_division=0))

    rec_macro = float(recall_score(y_true_clean, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true_clean, y_pred, average="weighted", zero_division=0))

    f1_mac = float(f1_score(y_true_clean, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true_clean, y_pred, average="weighted", zero_division=0))

    kappa = float(cohen_kappa_score(y_true_clean, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true_clean, y_pred, labels=classes if class_order else None)
    cm_list = cm.tolist()

    # Per-class metrics via classification_report
    report_dict = classification_report(
        y_true_clean,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for cls_name in classes:
        # Check if present in report_dict (might be formatted as str)
        key = cls_name if cls_name in report_dict else None
        if key is None:
            # Try type matching
            for k in report_dict:
                if str(k) == str(cls_name):
                    key = k
                    break
        if key and isinstance(report_dict[key], dict):
            per_class_metrics[cls_name] = {
                "precision": round(float(report_dict[key].get("precision", 0.0)), 4),
                "recall": round(float(report_dict[key].get("recall", 0.0)), 4),
                "f1_score": round(float(report_dict[key].get("f1-score", 0.0)), 4),
                "support": int(report_dict[key].get("support", 0)),
            }
        else:
            per_class_metrics[cls_name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "support": 0
            }

    # Optional ROC-AUC
    roc_auc_val: Optional[float] = None
    y_prob_list: Optional[List[List[float]]] = None
    try:
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)
            y_prob_list = y_prob.tolist()

            if len(classes) == 2:
                # Binary classification
                # Extract prob of positive class (second column or matching class)
                pos_idx = 1 if y_prob.shape[1] > 1 else 0
                roc_auc_val = float(roc_auc_score(y_true_clean, y_prob[:, pos_idx]))
            elif len(classes) > 2:
                # Multiclass OvR
                roc_auc_val = float(
                    roc_auc_score(y_true_clean, y_prob, multi_class="ovr", average="macro")
                )
    except Exception:
        # Gracefully handle estimators that fail probability scoring
        roc_auc_val = None

    # Extract feature importances if available (e.g. Random Forest)
    feat_importances: Optional[Dict[str, float]] = None
    try:
        classifier_step = pipeline.named_steps.get("classifier", None)
        if classifier_step is not None and hasattr(classifier_step, "feature_importances_"):
            importances = classifier_step.feature_importances_
            cols = list(X_test.columns)
            if len(importances) == len(cols):
                feat_importances = {
                    cols[i]: round(float(importances[i]), 4)
                    for i in range(len(cols))
                }
                # Sort descending
                feat_importances = dict(
                    sorted(feat_importances.items(), key=lambda item: item[1], reverse=True)
                )
    except Exception:
        feat_importances = None

    return EvaluationResult(
        model_id=model_id,
        model_name=model_name,
        hyperparameters=hyperparameters,
        is_success=True,
        train_time_sec=train_time_sec,
        predict_time_sec=predict_time_sec,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision_macro=prec_macro,
        precision_weighted=prec_weighted,
        recall_macro=rec_macro,
        recall_weighted=rec_weighted,
        f1_macro=f1_mac,
        f1_weighted=f1_wt,
        cohen_kappa=kappa,
        roc_auc=roc_auc_val,
        confusion_matrix=cm_list,
        classes=classes,
        per_class_metrics=per_class_metrics,
        y_pred=[str(p) for p in y_pred],
        y_prob=y_prob_list,
        feature_importances=feat_importances,
        trained_pipeline=pipeline
    )
