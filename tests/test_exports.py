"""
Tests for export and serialization functionality.
"""

import io
import json
import joblib
from classification.export import (
    export_classification_report_csv,
    export_comparison_csv,
    export_experiment_json,
    export_experiment_parameters_json,
    export_reproducible_python_code,
    serialize_trained_model,
)
from classification.sample_data import get_default_sample_bundle
from classification.training import train_and_evaluate


def test_export_formats():
    """Verify that CSV, JSON, joblib, and Python code exports function accurately."""
    bundle = get_default_sample_bundle(n_samples=150, n_features=8, n_classes=2)
    experiment = train_and_evaluate(bundle, ["lda", "rf"])

    # 1. Comparison CSV
    csv_str = export_comparison_csv(experiment)
    assert "Model,Accuracy,Balanced Accuracy" in csv_str
    assert "Linear Discriminant Analysis" in csv_str
    assert "Random Forest" in csv_str

    # 2. Experiment JSON
    json_str = export_experiment_json(experiment)
    parsed = json.loads(json_str)
    assert parsed["dataset_name"] == bundle.dataset_name
    assert "lda" in parsed["model_results"]
    assert "rf" in parsed["model_results"]
    assert parsed["n_train_samples"] == bundle.n_train_samples

    # 3. Parameters JSON
    params_json = export_experiment_parameters_json(experiment)
    params_data = json.loads(params_json)
    assert "model_hyperparameters" in params_data
    assert "lda" in params_data["model_hyperparameters"]

    # 4. Reproducible Python Code (.py)
    code_str = export_reproducible_python_code(experiment)
    assert "def create_model_pipeline" in code_str
    assert "def run_experiment" in code_str
    assert "HYPERPARAMETERS" in code_str

    # 5. Per-class report CSV
    report_csv = export_classification_report_csv(experiment.model_results["lda"])
    assert "Class,precision,recall,f1_score,support" in report_csv

    # 6. Joblib serialization
    model_bytes = serialize_trained_model(experiment.model_results["rf"])
    assert len(model_bytes) > 0
    # Deserialization test
    loaded_pipeline = joblib.load(io.BytesIO(model_bytes))
    test_pred = loaded_pipeline.predict(bundle.X_test.iloc[:5])
    assert len(test_pred) == 5
