"""
EEG Classification Lab - Core Machine Learning Package
Provides dataset schemas, model registry, validation, training pipeline, and evaluation metrics.
"""

from .schemas import DatasetBundle, EvaluationResult, ExperimentResult, ModelConfig
from .data_loader import prepare_dataset_from_single_df, prepare_dataset_from_separate_dfs, prepare_dataset_from_arrays
from .sample_data import generate_synthetic_eeg_dataset
from .model_registry import get_registered_models, get_model_definition
from .training import train_and_evaluate
from .validation import validate_dataset

__all__ = [
    "DatasetBundle",
    "EvaluationResult",
    "ExperimentResult",
    "ModelConfig",
    "prepare_dataset_from_single_df",
    "prepare_dataset_from_separate_dfs",
    "prepare_dataset_from_arrays",
    "generate_synthetic_eeg_dataset",
    "get_registered_models",
    "get_model_definition",
    "train_and_evaluate",
    "validate_dataset",
]
