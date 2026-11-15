"""
Data structures and schema definitions for the EEG Classification Lab.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    """
    Standard data container passed to the model training engine.
    Ensures strict separation between training and test sets.
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: List[str]
    class_names: List[str]
    dataset_name: str = "EEG Feature Dataset"
    source_type: str = "sample"  # 'sample', 'upload_single', 'upload_split', 'integrated'
    participant_train: Optional[pd.Series] = None
    participant_test: Optional[pd.Series] = None
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_train_samples(self) -> int:
        return len(self.X_train)

    @property
    def n_test_samples(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


@dataclass
class ModelConfig:
    """
    Configuration specification for a single machine learning model.
    """
    model_id: str
    model_name: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Evaluation metrics and state for a single trained classifier.
    """
    model_id: str
    model_name: str
    hyperparameters: Dict[str, Any]
    is_success: bool = True
    error_message: Optional[str] = None
    train_time_sec: float = 0.0
    predict_time_sec: float = 0.0

    # Core scalar metrics
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    precision_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_macro: float = 0.0
    recall_weighted: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    cohen_kappa: float = 0.0
    roc_auc: Optional[float] = None

    # Granular details
    confusion_matrix: Optional[List[List[int]]] = None
    classes: List[str] = field(default_factory=list)
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Predictions & feature importances
    y_pred: Optional[List[Any]] = None
    y_prob: Optional[List[List[float]]] = None
    feature_importances: Optional[Dict[str, float]] = None

    # Trained scikit-learn estimator / pipeline
    trained_pipeline: Any = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Returns a flat dictionary suitable for comparative tables."""
        return {
            "Model": self.model_name,
            "Accuracy": round(self.accuracy, 4),
            "Balanced Accuracy": round(self.balanced_accuracy, 4),
            "Macro F1": round(self.f1_macro, 4),
            "Weighted F1": round(self.f1_weighted, 4),
            "Macro Precision": round(self.precision_macro, 4),
            "Macro Recall": round(self.recall_macro, 4),
            "Cohen's Kappa": round(self.cohen_kappa, 4),
            "Train Time (s)": round(self.train_time_sec, 4),
            "Predict Time (s)": round(self.predict_time_sec, 4),
            "Status": "Success" if self.is_success else "Failed",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes full evaluation result excluding raw pipeline object."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "is_success": self.is_success,
            "error_message": self.error_message,
            "train_time_sec": self.train_time_sec,
            "predict_time_sec": self.predict_time_sec,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "precision_weighted": self.precision_weighted,
            "recall_macro": self.recall_macro,
            "recall_weighted": self.recall_weighted,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "cohen_kappa": self.cohen_kappa,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "classes": [str(c) for c in self.classes],
            "per_class_metrics": self.per_class_metrics,
            "feature_importances": self.feature_importances,
        }


@dataclass
class ExperimentResult:
    """
    Complete experiment record capturing dataset metadata and all trained models.
    """
    timestamp: str
    dataset_name: str
    source_type: str
    n_train_samples: int
    n_test_samples: int
    n_features: int
    feature_names: List[str]
    classes: List[str]
    model_results: Dict[str, EvaluationResult]
    total_duration_sec: float
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts model results into a summary DataFrame."""
        rows = []
        for result in self.model_results.values():
            if result.is_success:
                rows.append(result.to_summary_dict())
            else:
                rows.append({
                    "Model": result.model_name,
                    "Accuracy": np.nan,
                    "Balanced Accuracy": np.nan,
                    "Macro F1": np.nan,
                    "Weighted F1": np.nan,
                    "Macro Precision": np.nan,
                    "Macro Recall": np.nan,
                    "Cohen's Kappa": np.nan,
                    "Train Time (s)": np.nan,
                    "Predict Time (s)": np.nan,
                    "Status": f"Failed: {result.error_message}",
                })
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes full experiment results for JSON export."""
        return {
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "source_type": self.source_type,
            "is_synthetic": self.is_synthetic,
            "n_train_samples": self.n_train_samples,
            "n_test_samples": self.n_test_samples,
            "n_features": self.n_features,
            "classes": [str(c) for c in self.classes],
            "feature_names": self.feature_names,
            "total_duration_sec": round(self.total_duration_sec, 4),
            "metadata": self.metadata,
            "model_results": {
                m_id: res.to_dict() for m_id, res in self.model_results.items()
            },
        }
