"""
Dataset validation module for EEG feature matrices and target vectors.
Ensures rigorous data integrity, detects dimensionality mismatches, non-numeric values,
missing/infinite entries, and class distributions before training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Detailed summary of dataset checks, errors, and warnings."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


def validate_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    min_samples: int = 10
) -> ValidationReport:
    """
    Validates training and optional testing datasets against essential ML requirements.
    
    Checks:
    1. Dimension consistency (X_train rows == y_train rows, X_test rows == y_test rows).
    2. Feature count and column alignment between train and test.
    3. Non-numeric data detection.
    4. Missing (NaN) and infinite (Inf) values.
    5. Class distribution (at least 2 classes in y_train, single-class test set warnings).
    6. Minimum sample threshold.
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {}

    # --- 1. Check Training Data Presence ---
    if X_train is None or y_train is None:
        errors.append("Training features (X_train) and labels (y_train) must be provided.")
        return ValidationReport(is_valid=False, errors=errors)

    if not isinstance(X_train, pd.DataFrame):
        try:
            X_train = pd.DataFrame(X_train)
        except Exception as e:
            errors.append(f"Could not convert X_train to DataFrame: {str(e)}")
            return ValidationReport(is_valid=False, errors=errors)

    if not isinstance(y_train, (pd.Series, np.ndarray, list)):
        errors.append("y_train must be a pandas Series, NumPy array, or list.")
        return ValidationReport(is_valid=False, errors=errors)
    
    y_train = pd.Series(y_train)

    # --- 2. Check Row Counts Match ---
    if len(X_train) != len(y_train):
        errors.append(
            f"Row count mismatch in training data: X_train has {len(X_train)} samples, "
            f"but y_train has {len(y_train)} labels."
        )

    if len(X_train) < min_samples:
        errors.append(
            f"Training sample count ({len(X_train)}) is too small. "
            f"A minimum of {min_samples} samples is required."
        )

    # --- 3. Check Numeric Types in X_train ---
    non_numeric_train_cols = []
    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            non_numeric_train_cols.append(str(col))
    
    if non_numeric_train_cols:
        errors.append(
            f"X_train contains non-numeric column(s): {', '.join(non_numeric_train_cols[:5])} "
            f"{'...' if len(non_numeric_train_cols) > 5 else ''}. "
            "All feature columns must be numeric floats or integers."
        )

    # --- 4. Check Missing & Infinite Values in X_train ---
    try:
        nan_count_train = int(X_train.isna().sum().sum())
        if nan_count_train > 0:
            warnings.append(
                f"X_train contains {nan_count_train} missing (NaN) values. "
                "These will be imputed using median values inside the pipeline."
            )
        
        # Check infinity
        numeric_train = X_train.select_dtypes(include=[np.number])
        inf_count_train = int(np.isinf(numeric_train.values).sum()) if not numeric_train.empty else 0
        if inf_count_train > 0:
            errors.append(
                f"X_train contains {inf_count_train} infinite (Inf/-Inf) values. "
                "Infinite values cannot be processed; please clean or replace them."
            )
    except Exception as e:
        warnings.append(f"Error checking NaN/Inf in X_train: {str(e)}")

    # --- 5. Check Class Distribution in y_train ---
    nan_labels_train = int(y_train.isna().sum())
    if nan_labels_train > 0:
        errors.append(f"y_train contains {nan_labels_train} missing (NaN/None) labels.")

    clean_y_train = y_train.dropna()
    unique_classes_train = clean_y_train.unique()
    n_classes_train = len(unique_classes_train)
    
    if n_classes_train < 2:
        errors.append(
            f"y_train contains only {n_classes_train} unique class ({list(unique_classes_train)}). "
            "Classification requires at least 2 distinct classes."
        )
    
    train_class_counts = clean_y_train.value_counts().to_dict()
    info["train_class_counts"] = train_class_counts
    info["train_samples"] = len(X_train)
    info["n_features"] = X_train.shape[1]

    # Check for extreme class imbalance in train
    if n_classes_train >= 2:
        counts = list(train_class_counts.values())
        min_c, max_c = min(counts), max(counts)
        if max_c / max(min_c, 1) > 10:
            warnings.append(
                f"Severe class imbalance detected in training set (ratio {max_c}:{min_c}). "
                "Consider using balanced class weights or evaluating with Balanced Accuracy and Macro F1."
            )

    # --- 6. Check Test Data if Provided ---
    if X_test is not None:
        if not isinstance(X_test, pd.DataFrame):
            try:
                X_test = pd.DataFrame(X_test)
            except Exception as e:
                errors.append(f"Could not convert X_test to DataFrame: {str(e)}")
                return ValidationReport(is_valid=False, errors=errors)

        if y_test is None:
            errors.append("X_test was provided but y_test is missing.")
        else:
            y_test = pd.Series(y_test)
            if len(X_test) != len(y_test):
                errors.append(
                    f"Row count mismatch in test data: X_test has {len(X_test)} samples, "
                    f"but y_test has {len(y_test)} labels."
                )

        # Feature count consistency
        if X_test.shape[1] != X_train.shape[1]:
            errors.append(
                f"Feature dimension mismatch: X_train has {X_train.shape[1]} columns, "
                f"but X_test has {X_test.shape[1]} columns."
            )
        else:
            # Check column names if available
            train_cols = [str(c) for c in X_train.columns]
            test_cols = [str(c) for c in X_test.columns]
            if train_cols != test_cols:
                warnings.append(
                    "Feature column names in X_test do not strictly match X_train. "
                    "Features will be aligned by positional column index."
                )

        # Check numeric types in X_test
        non_numeric_test_cols = [
            str(col) for col in X_test.columns if not pd.api.types.is_numeric_dtype(X_test[col])
        ]
        if non_numeric_test_cols:
            errors.append(
                f"X_test contains non-numeric column(s): {', '.join(non_numeric_test_cols[:5])}."
            )

        # Check NaN and Inf in X_test
        nan_count_test = int(X_test.isna().sum().sum())
        if nan_count_test > 0:
            warnings.append(
                f"X_test contains {nan_count_test} missing (NaN) values. "
                "These will be imputed using median values computed from X_train."
            )

        numeric_test = X_test.select_dtypes(include=[np.number])
        inf_count_test = int(np.isinf(numeric_test.values).sum()) if not numeric_test.empty else 0
        if inf_count_test > 0:
            errors.append(
                f"X_test contains {inf_count_test} infinite (Inf/-Inf) values."
            )

        # Check test labels
        if y_test is not None:
            clean_y_test = y_test.dropna()
            test_classes = set(clean_y_test.unique())
            train_classes_set = set(unique_classes_train)
            unseen_classes = test_classes - train_classes_set
            if unseen_classes:
                errors.append(
                    f"Test set contains unseen classes not present in training data: {list(unseen_classes)}."
                )
            missing_classes = train_classes_set - test_classes
            if missing_classes:
                warnings.append(
                    f"Test set is missing classes present in training data: {list(missing_classes)}."
                )
            info["test_samples"] = len(X_test)
            info["test_class_counts"] = clean_y_test.value_counts().to_dict()

    is_valid = len(errors) == 0
    return ValidationReport(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        info=info
    )
