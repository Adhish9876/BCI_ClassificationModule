"""
Data loader and dataset preparation module.
Supports:
1. Single CSV with configurable train-test split or group-based participant-aware split.
2. Separate CSVs for X_train, y_train, X_test, y_test.
3. In-memory arrays / dataframes from the upstream Feature Extraction module.
"""

import io
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .schemas import DatasetBundle
from .validation import validate_dataset


def load_csv_data(file_input: Any) -> Union[pd.DataFrame, pd.Series]:
    """
    Safely reads CSV file, uploaded stream, DataFrame, or Series.
    """
    if isinstance(file_input, (pd.DataFrame, pd.Series)):
        return file_input.copy()
    
    if hasattr(file_input, "read"):
        # Uploaded file object from Streamlit or io.StringIO/BytesIO
        try:
            return pd.read_csv(file_input)
        except Exception:
            file_input.seek(0)
            return pd.read_csv(io.StringIO(file_input.read().decode("utf-8", errors="replace")))
    
    # Path string
    return pd.read_csv(file_input)


def prepare_dataset_from_single_df(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    group_column: Optional[str] = None,
    dataset_name: str = "Uploaded Dataset",
    feature_columns: Optional[List[str]] = None
) -> DatasetBundle:
    """
    Splits a single labelled DataFrame into X_train, y_train, X_test, y_test.
    
    If `group_column` is provided, performs a participant-aware GroupShuffleSplit
    so that trials from the same participant do not cross between train and test sets,
    preventing subject-identity data leakage.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns.")

    # Determine numeric feature columns
    if feature_columns is not None:
        feature_cols = [c for c in feature_columns if c in df.columns and c != target_column and c != group_column]
    else:
        # Exclude target, group column, and non-numeric metadata columns (like subject string IDs)
        feature_cols = [
            c for c in df.columns
            if c != target_column and c != group_column and pd.api.types.is_numeric_dtype(df[c])
        ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns available after removing target and metadata columns.")

    X = df[feature_cols].copy()
    y = df[target_column].copy()
    groups = df[group_column].copy() if (group_column and group_column in df.columns) else None

    # Perform group split if group_column is provided
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        part_train = groups.iloc[train_idx].copy()
        part_test = groups.iloc[test_idx].copy()
        split_method = f"Participant-aware GroupShuffleSplit on '{group_column}'"
    else:
        # Standard stratified train-test split
        strat = y if stratify else None
        split_res = train_test_split(
            X, y,
            test_size=test_size,
            stratify=strat,
            random_state=random_state
        )
        X_train, X_test, y_train, y_test = split_res
        part_train, part_test = None, None
        split_method = f"Standard {'Stratified ' if stratify else ''}Train-Test Split (test_size={test_size})"

    class_names = sorted([str(c) for c in y.unique()])
    feature_names = [str(c) for c in feature_cols]

    bundle = DatasetBundle(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=feature_names,
        class_names=class_names,
        dataset_name=dataset_name,
        source_type="upload_single",
        participant_train=part_train.reset_index(drop=True) if part_train is not None else None,
        participant_test=part_test.reset_index(drop=True) if part_test is not None else None,
        is_synthetic=False,
        metadata={
            "split_method": split_method,
            "test_size": test_size,
            "random_state": random_state,
            "group_column": group_column,
        }
    )

    # Validate bundle
    report = validate_dataset(bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test)
    if not report.is_valid:
        raise ValueError(f"Dataset validation failed: {'; '.join(report.errors)}")

    return bundle


def prepare_dataset_from_separate_dfs(
    X_train_in: Any,
    y_train_in: Any,
    X_test_in: Any,
    y_test_in: Any,
    dataset_name: str = "Uploaded Separate Sets"
) -> DatasetBundle:
    """
    Constructs a DatasetBundle from pre-split X_train, y_train, X_test, y_test CSVs/DataFrames/Series.
    """
    X_train = load_csv_data(X_train_in)
    X_test = load_csv_data(X_test_in)

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    # Labels can be a 1-column CSV, DataFrame, or Series
    raw_y_train = load_csv_data(y_train_in)
    if isinstance(raw_y_train, pd.DataFrame):
        y_train = raw_y_train.iloc[:, 0]
    else:
        y_train = pd.Series(raw_y_train)

    raw_y_test = load_csv_data(y_test_in)
    if isinstance(raw_y_test, pd.DataFrame):
        y_test = raw_y_test.iloc[:, 0]
    else:
        y_test = pd.Series(raw_y_test)

    # Validate before packaging
    report = validate_dataset(X_train, y_train, X_test, y_test)
    if not report.is_valid:
        raise ValueError(f"Validation failed for separate datasets: {'; '.join(report.errors)}")

    feature_names = [str(c) for c in X_train.columns]
    class_names = sorted([str(c) for c in y_train.unique()])

    return DatasetBundle(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=feature_names,
        class_names=class_names,
        dataset_name=dataset_name,
        source_type="upload_separate",
        is_synthetic=False,
        metadata={"source": "Pre-split separate CSV files"}
    )


def prepare_dataset_from_arrays(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    dataset_name: str = "Integrated Feature Extraction Data"
) -> DatasetBundle:
    """
    Direct in-memory integration interface designed to receive feature matrices
    produced by the upstream Feature Extraction module.
    """
    if isinstance(X_train, np.ndarray):
        cols = feature_names if feature_names and len(feature_names) == X_train.shape[1] else [f"Feature_{i+1}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=cols)
    else:
        X_train_df = X_train.copy()
        cols = list(X_train_df.columns)

    if isinstance(X_test, np.ndarray):
        X_test_df = pd.DataFrame(X_test, columns=cols)
    else:
        X_test_df = X_test.copy()

    y_train_s = pd.Series(y_train).copy()
    y_test_s = pd.Series(y_test).copy()

    report = validate_dataset(X_train_df, y_train_s, X_test_df, y_test_s)
    if not report.is_valid:
        raise ValueError(f"Validation failed for integrated dataset: {'; '.join(report.errors)}")

    detected_classes = sorted([str(c) for c in y_train_s.unique()])
    final_classes = class_names if class_names else detected_classes

    return DatasetBundle(
        X_train=X_train_df.reset_index(drop=True),
        y_train=y_train_s.reset_index(drop=True),
        X_test=X_test_df.reset_index(drop=True),
        y_test=y_test_s.reset_index(drop=True),
        feature_names=[str(c) for c in cols],
        class_names=final_classes,
        dataset_name=dataset_name,
        source_type="integrated",
        is_synthetic=False,
        metadata={"source": "Upstream Feature Extraction Module Interface"}
    )
