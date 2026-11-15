"""
Tests for training pipeline execution and multi-model training.
"""

from classification.sample_data import get_default_sample_bundle
from classification.training import train_and_evaluate


def test_train_all_five_primary_models():
    """Verify all 5 primary classifiers train successfully and produce valid results."""
    bundle = get_default_sample_bundle(n_samples=200, n_features=10, n_classes=2, test_size=0.25)
    selected = ["lda", "svm", "rf", "knn", "lr"]
    
    experiment = train_and_evaluate(bundle, selected)
    
    assert experiment.n_train_samples == 150
    assert experiment.n_test_samples == 50
    assert len(experiment.model_results) == 5

    for m_id in selected:
        res = experiment.model_results[m_id]
        assert res.is_success, f"Model {m_id} failed: {res.error_message}"
        assert 0.0 <= res.accuracy <= 1.0
        assert 0.0 <= res.f1_macro <= 1.0
        assert res.train_time_sec >= 0.0
        assert res.predict_time_sec >= 0.0
        assert res.trained_pipeline is not None
        assert res.confusion_matrix is not None


def test_train_multiclass_three_classes():
    """Verify training on 3-class EEG feature dataset."""
    bundle = get_default_sample_bundle(n_samples=240, n_features=12, n_classes=3, test_size=0.25)
    selected = ["lda", "svm", "rf", "knn", "lr"]

    experiment = train_and_evaluate(bundle, selected)
    assert len(experiment.classes) == 3

    for m_id in selected:
        res = experiment.model_results[m_id]
        assert res.is_success
        assert len(res.confusion_matrix) == 3
        assert len(res.confusion_matrix[0]) == 3


def test_partial_failure_resilience():
    """Verify that an invalid configuration on one model does not fail the remaining models."""
    bundle = get_default_sample_bundle(n_samples=150, n_features=8, n_classes=2)
    selected = ["svm", "lda"]
    
    # Pass an invalid parameter for SVM that will fail fit or pipeline creation
    configs = {
        "svm": {"C": -5.0}  # C <= 0 is invalid in sklearn
    }

    experiment = train_and_evaluate(bundle, selected, model_configs=configs)
    
    # SVM should fail gracefully
    assert not experiment.model_results["svm"].is_success
    assert experiment.model_results["svm"].error_message is not None

    # LDA must still succeed
    assert experiment.model_results["lda"].is_success
    assert experiment.model_results["lda"].accuracy > 0.0
