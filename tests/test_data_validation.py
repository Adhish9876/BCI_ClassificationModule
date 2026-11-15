"""
Tests for dataset validation, loading, and schema consistency.
"""

import numpy as np
import pandas as pd
import pytest

from classification.data_loader import (
    prepare_dataset_from_arrays,
    prepare_dataset_from_separate_dfs,
    prepare_dataset_from_single_df,
)
from classification.sample_data import generate_synthetic_eeg_dataframe, get_default_sample_bundle
from classification.validation import validate_dataset


def test_sample_dataset_generation():
    """Verify default synthetic EEG generator produces valid dimensions and metadata."""
    bundle = get_default_sample_bundle(n_samples=200, n_features=12, n_classes=2, test_size=0.25)
    assert bundle.n_train_samples == 150
    assert bundle.n_test_samples == 50
    assert bundle.n_features == 12
    assert bundle.n_classes == 2
    assert bundle.is_synthetic is True
    assert len(bundle.feature_names) == 12
    assert bundle.participant_train is not None
    assert bundle.participant_test is not None


def test_validation_dimension_mismatch():
    """Test detection of mismatched sample counts between X and y."""
    X = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series([0, 1] * 20)  # Length 40 != 50
    report = validate_dataset(X, y)
    assert not report.is_valid
    assert any("mismatch" in err.lower() for err in report.errors)


def test_validation_invalid_feature_dimensions():
    """Test detection of inconsistent feature columns between train and test."""
    X_train = pd.DataFrame(np.random.randn(40, 5), columns=[f"f{i}" for i in range(5)])
    y_train = pd.Series([0, 1] * 20)
    X_test = pd.DataFrame(np.random.randn(20, 4), columns=[f"f{i}" for i in range(4)])
    y_test = pd.Series([0, 1] * 10)

    report = validate_dataset(X_train, y_train, X_test, y_test)
    assert not report.is_valid
    assert any("dimension mismatch" in err.lower() for err in report.errors)


def test_validation_non_numeric_features():
    """Test detection of string/object columns in features."""
    X = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": ["bad", "values", "here", "too"]
    })
    y = pd.Series([0, 1, 0, 1])
    report = validate_dataset(X, y)
    assert not report.is_valid
    assert any("non-numeric" in err.lower() for err in report.errors)


def test_validation_missing_values_warning():
    """Test that missing values produce an informative warning (since pipeline can impute them)."""
    X = pd.DataFrame(np.random.randn(40, 4), columns=["f1", "f2", "f3", "f4"])
    X.iloc[2, 1] = np.nan
    y = pd.Series([0, 1] * 20)
    report = validate_dataset(X, y)
    assert report.is_valid  # Still valid because imputer is in pipeline
    assert any("missing" in w.lower() for w in report.warnings)


def test_validation_single_class_error():
    """Test error when training data has fewer than 2 classes."""
    X = pd.DataFrame(np.random.randn(30, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series([1] * 30)  # Only 1 class
    report = validate_dataset(X, y)
    assert not report.is_valid
    assert any("at least 2 distinct classes" in err.lower() for err in report.errors)


def test_single_df_train_test_split():
    """Test splitting a single DataFrame with stratification."""
    df, target_col, part_col = generate_synthetic_eeg_dataframe(n_samples=100, n_features=8, n_classes=2)
    bundle = prepare_dataset_from_single_df(df, target_column=target_col, test_size=0.3, random_state=42)
    assert bundle.n_train_samples == 70
    assert bundle.n_test_samples == 30
    assert bundle.n_features == 8


def test_participant_aware_group_split():
    """Test participant-aware splitting prevents subject overlap between train and test."""
    df, target_col, part_col = generate_synthetic_eeg_dataframe(n_samples=120, n_features=6, n_participants=6)
    bundle = prepare_dataset_from_single_df(
        df,
        target_column=target_col,
        test_size=0.33,
        group_column=part_col,
        random_state=42
    )
    train_subjects = set(bundle.participant_train.unique())
    test_subjects = set(bundle.participant_test.unique())
    # Crucial assertion: ZERO subject overlap!
    assert len(train_subjects.intersection(test_subjects)) == 0


def test_separate_dfs_loader():
    """Test loading pre-split separate DataFrames."""
    X_tr = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y_tr = pd.Series([0, 1] * 25)
    X_te = pd.DataFrame(np.random.randn(20, 4), columns=["f1", "f2", "f3", "f4"])
    y_te = pd.Series([0, 1] * 10)

    bundle = prepare_dataset_from_separate_dfs(X_tr, y_tr, X_te, y_te)
    assert bundle.n_train_samples == 50
    assert bundle.n_test_samples == 20
    assert bundle.n_features == 4


def test_integrated_arrays_loader():
    """Test receiving in-memory numpy arrays from upstream module."""
    X_tr = np.random.randn(60, 5)
    y_tr = np.array([0, 1] * 30)
    X_te = np.random.randn(20, 5)
    y_te = np.array([0, 1] * 10)

    bundle = prepare_dataset_from_arrays(X_tr, y_tr, X_te, y_te, feature_names=["c1", "c2", "c3", "c4", "c5"])
    assert bundle.n_train_samples == 60
    assert bundle.n_test_samples == 20
    assert bundle.n_features == 5
