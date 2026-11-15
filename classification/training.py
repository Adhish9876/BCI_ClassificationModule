"""
Central training orchestration engine for the EEG Classification Lab.
Executes training pipelines across selected models on identical data, records execution times,
and aggregates comparative metrics into an ExperimentResult container.
"""

from datetime import datetime
import time
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from .evaluation import evaluate_classifier
from .model_registry import get_model_definition
from .schemas import DatasetBundle, EvaluationResult, ExperimentResult
from .validation import validate_dataset


def train_and_evaluate(
    dataset: DatasetBundle,
    selected_models: List[str],
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> ExperimentResult:
    """
    Core training and evaluation execution function.
    
    Parameters
    ----------
    dataset : DatasetBundle
        Container with X_train, y_train, X_test, y_test.
    selected_models : List[str]
        List of model identifiers to train (e.g. ['lda', 'svm', 'rf', 'knn', 'lr']).
    model_configs : Optional[Dict[str, Dict[str, Any]]]
        Hyperparameter overrides per model_id.
    progress_callback : Optional[Callable[[int, int, str], None]]
        Callback for UI progress tracking (current_index, total_count, model_name).

    Returns
    -------
    ExperimentResult
        Comprehensive result object containing evaluated metrics for each classifier.
    """
    if model_configs is None:
        model_configs = {}

    # Validate dataset before starting
    val_report = validate_dataset(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)
    if not val_report.is_valid:
        raise ValueError(f"Dataset validation failed: {'; '.join(val_report.errors)}")

    if not selected_models:
        raise ValueError("At least one classifier must be selected for training.")

    total_start = time.perf_counter()
    results: Dict[str, EvaluationResult] = {}
    total_models = len(selected_models)

    for idx, model_id in enumerate(selected_models):
        model_def = get_model_definition(model_id)
        if model_def is None:
            results[model_id] = EvaluationResult(
                model_id=model_id,
                model_name=model_id.upper(),
                hyperparameters={},
                is_success=False,
                error_message=f"Model identifier '{model_id}' is not recognized in registry."
            )
            continue

        if progress_callback:
            progress_callback(idx + 1, total_models, model_def.name)

        params = model_configs.get(model_id, {})

        # Handle PyRiemann covariance matrix constraint
        if model_def.requires_covariance:
            # If input is 2D flat feature vector, check if it can be used or skip with explanatory error
            if dataset.X_train.ndim == 2:
                results[model_id] = EvaluationResult(
                    model_id=model_id,
                    model_name=model_def.name,
                    hyperparameters=params,
                    is_success=False,
                    error_message=(
                        "Riemannian MDM requires 3D covariance matrices ([N_trials, Channels, Channels]). "
                        "Current input consists of standard 2D flat feature vectors."
                    )
                )
                continue

        # Build pipeline and execute training
        try:
            pipeline = model_def.create_pipeline(params)
            
            # Fit on training data ONLY (prevents data leakage)
            start_train = time.perf_counter()
            pipeline.fit(dataset.X_train, dataset.y_train)
            train_duration = time.perf_counter() - start_train

            # Evaluate on unseen test data
            eval_res = evaluate_classifier(
                pipeline=pipeline,
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                model_id=model_id,
                model_name=model_def.name,
                hyperparameters=params,
                train_time_sec=train_duration,
                class_order=dataset.class_names
            )
            results[model_id] = eval_res

        except Exception as e:
            # Catch individual model failures so other models are not discarded
            results[model_id] = EvaluationResult(
                model_id=model_id,
                model_name=model_def.name,
                hyperparameters=params,
                is_success=False,
                error_message=str(e)
            )

    total_duration = time.perf_counter() - total_start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return ExperimentResult(
        timestamp=timestamp,
        dataset_name=dataset.dataset_name,
        source_type=dataset.source_type,
        n_train_samples=dataset.n_train_samples,
        n_test_samples=dataset.n_test_samples,
        n_features=dataset.n_features,
        feature_names=dataset.feature_names,
        classes=dataset.class_names,
        model_results=results,
        total_duration_sec=total_duration,
        is_synthetic=dataset.is_synthetic,
        metadata=dataset.metadata
    )
