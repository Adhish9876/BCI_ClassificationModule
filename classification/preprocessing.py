"""
Preprocessing and scikit-learn Pipeline construction module.
Guarantees strict prevention of data leakage by encapsulating imputation
and feature standardization exclusively within estimator pipelines fitted only on training data.
"""

from typing import Any, Optional
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def build_pipeline(
    classifier: BaseEstimator,
    scale_features: bool = True,
    impute_missing: bool = True,
    scaler_type: str = "standard"  # 'standard', 'robust', 'minmax', or 'none'
) -> Pipeline:
    """
    Constructs a leakage-free scikit-learn Pipeline.
    
    Parameters
    ----------
    classifier : BaseEstimator
        The un-fitted base classifier.
    scale_features : bool, default=True
        Whether to include feature standardization. Essential for distance-based
        (KNN) and margin-based (SVM) algorithms, and gradient-based (Logistic Regression).
    impute_missing : bool, default=True
        Whether to impute missing feature entries with median values learned from training data.
    scaler_type : str, default='standard'
        'standard' for StandardScaler (zero mean, unit variance),
        'robust' for RobustScaler (median and interquartile range, resilient to EEG artifact outliers),
        'minmax' for MinMaxScaler (scales features into [0, 1] range), or
        'none' to bypass scaling.
        
    Returns
    -------
    Pipeline
        Configured Pipeline ready to fit on training data only.
    """
    steps = []

    if impute_missing:
        steps.append(("imputer", SimpleImputer(strategy="median")))

    if scale_features and str(scaler_type).lower() not in ("none", "raw", "false"):
        stype = str(scaler_type).lower()
        if stype == "robust":
            steps.append(("scaler", RobustScaler()))
        elif stype == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        else:
            steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", classifier))

    return Pipeline(steps)
