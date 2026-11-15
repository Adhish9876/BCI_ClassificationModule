"""
Tests for metric calculations, predictions, and evaluation outputs.
"""

from classification.sample_data import get_default_sample_bundle
from classification.training import train_and_evaluate


def test_metric_calculations_and_bounds():
    """Verify that all metrics are calculated within proper statistical bounds."""
    bundle = get_default_sample_bundle(n_samples=180, n_features=8, n_classes=2, test_size=0.25)
    experiment = train_and_evaluate(bundle, ["svm", "rf"])

    for m_id in ["svm", "rf"]:
        res = experiment.model_results[m_id]
        assert 0.0 <= res.accuracy <= 1.0
        assert 0.0 <= res.balanced_accuracy <= 1.0
        assert 0.0 <= res.precision_macro <= 1.0
        assert 0.0 <= res.recall_macro <= 1.0
        assert 0.0 <= res.f1_macro <= 1.0
        assert 0.0 <= res.f1_weighted <= 1.0
        assert -1.0 <= res.cohen_kappa <= 1.0

        # Check confusion matrix dimensions
        assert len(res.confusion_matrix) == 2
        assert len(res.confusion_matrix[0]) == 2
        total_cm_samples = sum(sum(row) for row in res.confusion_matrix)
        assert total_cm_samples == bundle.n_test_samples

        # Check per-class metrics
        assert len(res.per_class_metrics) == 2
        for cls_name, pcm in res.per_class_metrics.items():
            assert "precision" in pcm
            assert "recall" in pcm
            assert "f1_score" in pcm
            assert "support" in pcm
            assert pcm["support"] > 0


def test_rf_feature_importances():
    """Verify Random Forest extracts feature importances matching feature count."""
    bundle = get_default_sample_bundle(n_samples=160, n_features=12, n_classes=2)
    experiment = train_and_evaluate(bundle, ["rf"])
    rf_res = experiment.model_results["rf"]

    assert rf_res.feature_importances is not None
    assert len(rf_res.feature_importances) == 12
    # Verify importance sum is approximately 1.0
    total_imp = sum(rf_res.feature_importances.values())
    assert 0.95 <= total_imp <= 1.05
