"""
Tests for model registry, parameter validation, and pipeline creation.
"""

from classification.model_registry import get_model_definition, get_registered_models


def test_registered_primary_models():
    """Verify all 5 primary classifiers are registered."""
    models = get_registered_models(include_optional=False)
    for model_id in ["lda", "svm", "rf", "knn", "lr"]:
        assert model_id in models
        defn = models[model_id]
        assert defn.name is not None
        assert defn.short_description is not None
        assert defn.theory_summary is not None
        assert defn.bci_relevance is not None


def test_lda_svd_shrinkage_sanitization():
    """Verify that LDA with SVD solver sanitizes shrinkage to None (preventing sklearn error)."""
    lda_def = get_model_definition("lda")
    pipeline = lda_def.create_pipeline({"solver": "svd", "shrinkage": "auto"})
    lda_step = pipeline.named_steps["classifier"]
    assert lda_step.solver == "svd"
    assert lda_step.shrinkage is None


def test_lda_lsqr_shrinkage_valid():
    """Verify that LDA with LSQR solver accepts shrinkage."""
    lda_def = get_model_definition("lda")
    pipeline = lda_def.create_pipeline({"solver": "lsqr", "shrinkage": "auto"})
    lda_step = pipeline.named_steps["classifier"]
    assert lda_step.solver == "lsqr"
    assert lda_step.shrinkage == "auto"


def test_logistic_regression_solver_penalty_enforcement():
    """Verify that Logistic Regression enforces solver-penalty compatibility."""
    lr_def = get_model_definition("lr")
    # If user requests l1 with lbfgs (which is invalid in sklearn), our factory adapts solver to saga or liblinear
    pipeline = lr_def.create_pipeline({"penalty": "l1", "solver": "lbfgs"})
    lr_step = pipeline.named_steps["classifier"]
    assert lr_step.penalty == "l1"
    assert lr_step.solver in ("saga", "liblinear")


def test_pipeline_scaling_inclusion():
    """Verify standard scaler is included in SVM and KNN pipelines."""
    svm_def = get_model_definition("svm")
    svm_pipe = svm_def.create_pipeline({})
    assert "scaler" in svm_pipe.named_steps

    knn_def = get_model_definition("knn")
    knn_pipe = knn_def.create_pipeline({})
    assert "scaler" in knn_pipe.named_steps
