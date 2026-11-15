"""
Model registry and definition module for EEG Classification Lab.
Contains factory functions, parameter validation, descriptions, and educational explanations
for all supported BCI machine learning algorithms.
"""

from typing import Any, Callable, Dict, List, Optional
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import build_pipeline

# Check PyRiemann availability
try:
    import pyriemann
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False


class ModelDefinition:
    """Metadata, educational theory, and constructor for a classifier."""
    def __init__(
        self,
        model_id: str,
        name: str,
        category: str,
        short_description: str,
        theory_summary: str,
        bci_relevance: str,
        default_hyperparameters: Dict[str, Any],
        factory_fn: Callable[[Dict[str, Any]], Pipeline],
        requires_covariance: bool = False,
        is_optional_extension: bool = False
    ):
        self.model_id = model_id
        self.name = name
        self.category = category
        self.short_description = short_description
        self.theory_summary = theory_summary
        self.bci_relevance = bci_relevance
        self.default_hyperparameters = default_hyperparameters
        self.factory_fn = factory_fn
        self.requires_covariance = requires_covariance
        self.is_optional_extension = is_optional_extension

    def create_pipeline(self, user_params: Optional[Dict[str, Any]] = None) -> Pipeline:
        params = self.default_hyperparameters.copy()
        if user_params:
            params.update(user_params)
        return self.factory_fn(params)


# --- 1. Linear Discriminant Analysis Factory ---
def _create_lda_pipeline(params: Dict[str, Any]) -> Pipeline:
    solver = params.get("solver", "svd")
    shrinkage = params.get("shrinkage", None)
    scale_features = params.get("scale_features", True)
    scaler_type = params.get("scaler_type", "standard")

    # Validate solver & shrinkage compatibility
    if solver == "svd":
        # SVD does not support shrinkage in scikit-learn
        shrinkage = None
    elif shrinkage == "auto":
        shrinkage = "auto"
    elif isinstance(shrinkage, (float, int)):
        shrinkage = float(np_clip(shrinkage, 0.0, 1.0))
    else:
        shrinkage = None

    estimator = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    # LDA is robust, but scaling helps numerical stability for lsqr/eigen
    return build_pipeline(estimator, scale_features=scale_features, scaler_type=scaler_type)


# --- 2. Support Vector Machine Factory ---
def _create_svm_pipeline(params: Dict[str, Any]) -> Pipeline:
    kernel = params.get("kernel", "rbf")
    C = float(params.get("C", 1.0))
    gamma = params.get("gamma", "scale")
    degree = int(params.get("degree", 3))
    random_state = int(params.get("random_state", 42))
    scale_features = params.get("scale_features", True)
    scaler_type = params.get("scaler_type", "standard")

    estimator = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=True,  # Enables predict_proba for ROC-AUC and calibration
        random_state=random_state
    )
    # Feature scaling is mandatory for SVM
    return build_pipeline(estimator, scale_features=scale_features, scaler_type=scaler_type)


# --- 3. Random Forest Factory ---
def _create_rf_pipeline(params: Dict[str, Any]) -> Pipeline:
    n_estimators = int(params.get("n_estimators", 100))
    max_depth = params.get("max_depth", None)
    if max_depth is not None and max_depth != "None" and max_depth != "":
        max_depth = int(max_depth)
    else:
        max_depth = None

    min_samples_split = int(params.get("min_samples_split", 2))
    class_weight = params.get("class_weight", None)
    if class_weight in ("None", "", None):
        class_weight = None

    random_state = int(params.get("random_state", 42))
    scale_features = params.get("scale_features", False)
    scaler_type = params.get("scaler_type", "standard")

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    # Tree ensembles are scale-invariant, but pipeline handles potential NaNs cleanly
    return build_pipeline(estimator, scale_features=scale_features, scaler_type=scaler_type)


# --- 4. K-Nearest Neighbors Factory ---
def _create_knn_pipeline(params: Dict[str, Any]) -> Pipeline:
    n_neighbors = int(params.get("n_neighbors", 5))
    metric = params.get("metric", "euclidean")
    weights = params.get("weights", "uniform")
    scale_features = params.get("scale_features", True)
    scaler_type = params.get("scaler_type", "standard")

    estimator = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        n_jobs=-1
    )
    # Feature scaling is mandatory for distance-based KNN
    return build_pipeline(estimator, scale_features=scale_features, scaler_type=scaler_type)


# --- 5. Logistic Regression Factory ---
def _create_lr_pipeline(params: Dict[str, Any]) -> Pipeline:
    C = float(params.get("C", 1.0))
    solver = params.get("solver", "lbfgs")
    penalty = params.get("penalty", "l2")
    class_weight = params.get("class_weight", None)
    if class_weight in ("None", "", None):
        class_weight = None

    random_state = int(params.get("random_state", 42))
    scale_features = params.get("scale_features", True)
    scaler_type = params.get("scaler_type", "standard")

    # Enforce solver / penalty compatibility rules in scikit-learn
    if penalty == "l1" and solver not in ("liblinear", "saga"):
        solver = "saga"
    elif penalty == "elasticnet":
        solver = "saga"
    elif penalty in ("none", None) and solver not in ("lbfgs", "newton-cg", "saga"):
        solver = "lbfgs"

    estimator = LogisticRegression(
        C=C,
        solver=solver,
        penalty=penalty if penalty != "None" else None,
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state
    )
    # Feature scaling speeds up convergence and ensures regularizer acts equally
    return build_pipeline(estimator, scale_features=scale_features, scaler_type=scaler_type)


# --- 6. Optional PyRiemann MDM Factory ---
def _create_riemann_mdm_pipeline(params: Dict[str, Any]) -> Pipeline:
    metric = params.get("metric", "riemann")
    if not PYRIEMANN_AVAILABLE:
        raise RuntimeError("PyRiemann is not installed. MDM classifier is unavailable.")
    
    # MDM operates directly on covariance matrices
    mdm = MDM(metric=metric)
    # We return pipeline with MDM
    from sklearn.pipeline import Pipeline as SkPipeline
    return SkPipeline([("mdm", mdm)])


def np_clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


# Master Registry
MODEL_REGISTRY: Dict[str, ModelDefinition] = {
    "lda": ModelDefinition(
        model_id="lda",
        name="Linear Discriminant Analysis (LDA)",
        category="Linear Discriminative",
        short_description="Finds linear combination of features that maximizes separation between class centroids relative to within-class spread.",
        theory_summary=(
            "LDA projects data onto a lower-dimensional subspace by maximizing Fisher's criterion: "
            "the ratio of between-class scatter to within-class scatter. It assumes each class follows a multivariate "
            "Gaussian distribution with identical covariance matrices across classes."
        ),
        bci_relevance=(
            "LDA is the gold-standard baseline in BCI research (especially for P300 and Common Spatial Pattern SMR classification). "
            "Its closed-form solution trains in milliseconds, is immune to random seed variation, and prevents overfitting "
            "when trials are scarce."
        ),
        default_hyperparameters={
            "solver": "svd",
            "shrinkage": "None"
        },
        factory_fn=_create_lda_pipeline
    ),
    "svm": ModelDefinition(
        model_id="svm",
        name="Support Vector Machine (SVM)",
        category="Kernel / Maximum Margin",
        short_description="Finds the optimal separating hyperplane that maximizes the geometric margin to the nearest support vectors.",
        theory_summary=(
            "SVM seeks a decision boundary that maximizes the margin of separation between classes. "
            "Only data points lying on the margin boundary (the support vectors) define the solution. "
            "Nonlinear relationships are handled seamlessly using kernel tricks (RBF, Polynomial) which implicitly map "
            "features into higher-dimensional reproducing kernel Hilbert spaces."
        ),
        bci_relevance=(
            "Widely utilized across motor imagery, emotion recognition, and cognitive workload estimation. "
            "RBF kernels excel when EEG feature spaces have complex nonlinear structures, while linear SVM provides "
            "interpretable channel weights."
        ),
        default_hyperparameters={
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "degree": 3
        },
        factory_fn=_create_svm_pipeline
    ),
    "rf": ModelDefinition(
        model_id="rf",
        name="Random Forest",
        category="Ensemble / Decision Trees",
        short_description="Builds an ensemble of decorrelated decision trees using bootstrap aggregation and random feature subsets.",
        theory_summary=(
            "Random Forest constructs numerous randomized decision trees during training. Individual trees have high variance "
            "and low bias; by aggregating their predictions via majority voting (bagging), the overall ensemble achieves dramatically "
            "lower variance and superior generalization without overfitting."
        ),
        bci_relevance=(
            "Naturally robust to monotonic transformations and uncalibrated electrode gains. "
            "Importantly, Random Forest outputs Gini/MDI feature importances, allowing students to inspect which EEG bands "
            "or electrode channels contribute most to discrimination."
        ),
        default_hyperparameters={
            "n_estimators": 100,
            "max_depth": "None",
            "min_samples_split": 2,
            "class_weight": "None"
        },
        factory_fn=_create_rf_pipeline
    ),
    "knn": ModelDefinition(
        model_id="knn",
        name="K-Nearest Neighbors (KNN)",
        category="Instance-based / Non-parametric",
        short_description="Classifies unseen test samples based on majority class labels among their K closest training neighbors.",
        theory_summary=(
            "KNN is a non-parametric instance-based learner. It stores all training instances and makes predictions locally: "
            "for a new sample, it computes pairwise geometric distance to all stored points, retrieves the K nearest points, "
            "and takes a majority vote. The parameter K controls the bias-variance tradeoff: small K gives flexible but noise-sensitive "
            "boundaries; large K produces smoother boundaries."
        ),
        bci_relevance=(
            "Provides an intuitive geometric baseline for students to grasp distance metrics in feature space. "
            "However, its sensitivity to the curse of dimensionality and EEG trial outliers makes feature normalization "
            "and careful K tuning essential."
        ),
        default_hyperparameters={
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform"
        },
        factory_fn=_create_knn_pipeline
    ),
    "lr": ModelDefinition(
        model_id="lr",
        name="Logistic Regression",
        category="Linear Probabilistic",
        short_description="Models posterior class probabilities via logistic sigmoid transformation of linear feature combinations.",
        theory_summary=(
            "Logistic Regression optimizes log-likelihood to model class probabilities. Regularization penalties "
            "(L1 for sparsity and feature selection, L2 for coefficient shrinkage) prevent weights from inflating on noisy features. "
            "Its coefficients directly convey the log-odds impact of each extracted EEG feature."
        ),
        bci_relevance=(
            "Offers well-calibrated posterior probabilities essential for adaptive BCI systems, asynchronous brain switches, "
            "and confidence thresholding. L1 regularization acts as an automatic feature selector identifying key frequency bands."
        ),
        default_hyperparameters={
            "C": 1.0,
            "solver": "lbfgs",
            "penalty": "l2",
            "class_weight": "None"
        },
        factory_fn=_create_lr_pipeline
    )
}

# Optional PyRiemann MDM extension
if PYRIEMANN_AVAILABLE:
    MODEL_REGISTRY["riemann_mdm"] = ModelDefinition(
        model_id="riemann_mdm",
        name="Riemannian Minimum Distance to Mean (MDM)",
        category="Riemannian Geometry",
        short_description="Classifies covariance matrices directly on the Riemannian manifold of Symmetric Positive Definite (SPD) matrices.",
        theory_summary=(
            "Riemannian MDM projects trials onto the manifold of SPD covariance matrices using Affine-Invariant Riemannian Metric (AIRM). "
            "It computes the geometric Fréchet mean covariance matrix for each class and classifies new trials by geodesic distance."
        ),
        bci_relevance=(
            "State-of-the-art paradigm in modern BCI competitions (e.g. BCI Competition IV, Brain Invaders). "
            "Requires covariance matrix input (shape [N, channels, channels]), bypassing Euclidean distortions."
        ),
        default_hyperparameters={
            "metric": "riemann"
        },
        factory_fn=_create_riemann_mdm_pipeline,
        requires_covariance=True,
        is_optional_extension=True
    )


def get_registered_models(include_optional: bool = True) -> Dict[str, ModelDefinition]:
    """Returns dictionary of available models."""
    if include_optional:
        return MODEL_REGISTRY.copy()
    return {k: v for k, v in MODEL_REGISTRY.items() if not v.is_optional_extension}


def get_model_definition(model_id: str) -> Optional[ModelDefinition]:
    """Retrieves definition by model_id."""
    return MODEL_REGISTRY.get(model_id)
