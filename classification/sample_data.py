"""
Deterministic synthetic EEG feature dataset generator for educational and demonstration purposes.
Clearly identified as synthetic demonstration data (not actual EEG recordings).
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from .schemas import DatasetBundle

# Simulated EEG channel and frequency-band feature names
SIMULATED_EEG_FEATURE_NAMES = [
    "Mu_C3", "Mu_C4", "Beta_C3", "Beta_C4",
    "Alpha_O1", "Alpha_O2", "Alpha_P3", "Alpha_P4",
    "Theta_Fz", "Theta_Cz", "Beta_F3", "Beta_F4",
    "Gamma_T7", "Gamma_T8", "Alpha_Cz", "Beta_Pz",
    "Delta_Fp1", "Delta_Fp2", "Alpha_Fz", "Beta_Oz",
    "Theta_Pz", "Gamma_F3", "Gamma_F4", "Mu_Cz"
]

CLASS_LABEL_PRESETS = {
    2: ["Resting State", "Motor Imagery"],
    3: ["Rest", "Left Hand Imagery", "Right Hand Imagery"]
}


def generate_synthetic_eeg_dataframe(
    n_samples: int = 800,
    n_features: int = 16,
    n_classes: int = 2,
    n_participants: int = 8,
    random_state: int = 42,
    noise: float = 0.05
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """
    Generates a realistic synthetic EEG feature dataset as a single pandas DataFrame.
    
    Includes simulated EEG band power features, class targets, and participant IDs
    to allow testing participant-aware group-based splitting.
    """
    if n_features < 4 or n_features > len(SIMULATED_EEG_FEATURE_NAMES):
        n_features = min(max(n_features, 4), len(SIMULATED_EEG_FEATURE_NAMES))
    
    if n_classes not in (2, 3):
        n_classes = 2

    # Informative vs redundant feature allocation
    n_informative = max(2, int(n_features * 0.65))
    n_redundant = max(1, int(n_features * 0.20))
    n_repeated = 0
    
    # Ensure n_informative + n_redundant < n_features
    if n_informative + n_redundant >= n_features:
        n_informative = n_features - 2
        n_redundant = 1

    X_raw, y_raw = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=None,
        flip_y=noise,
        random_state=random_state
    )

    feature_cols = SIMULATED_EEG_FEATURE_NAMES[:n_features]
    df = pd.DataFrame(X_raw, columns=feature_cols)

    # Map class integers to descriptive EEG labels
    label_names = CLASS_LABEL_PRESETS[n_classes]
    df["Target_Class"] = [label_names[y] for y in y_raw]

    # Assign synthetic participant IDs (epochs grouped in consecutive blocks per subject)
    # Simulates continuous recording sessions where consecutive trials belong to the same subject
    samples_per_subj = n_samples // n_participants
    subj_ids = []
    for i in range(n_participants):
        count = samples_per_subj if i < n_participants - 1 else (n_samples - len(subj_ids))
        subj_ids.extend([f"Participant_{i+1:02d}"] * count)
    df["Participant_ID"] = subj_ids

    return df, "Target_Class", "Participant_ID"


def get_default_sample_bundle(
    n_samples: int = 800,
    n_features: int = 16,
    n_classes: int = 2,
    test_size: float = 0.2,
    random_state: int = 42
) -> DatasetBundle:
    """
    Constructs a ready-to-train DatasetBundle from the synthetic EEG generator.
    """
    df, target_col, part_col = generate_synthetic_eeg_dataframe(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state
    )

    feature_cols = [c for c in df.columns if c not in (target_col, part_col)]
    X = df[feature_cols]
    y = df[target_col]
    part = df[part_col]

    X_train, X_test, y_train, y_test, part_train, part_test = train_test_split(
        X, y, part,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return DatasetBundle(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=feature_cols,
        class_names=sorted(list(y.unique().astype(str))),
        dataset_name="Synthetic Demonstration EEG Features",
        source_type="sample",
        participant_train=part_train.reset_index(drop=True),
        participant_test=part_test.reset_index(drop=True),
        is_synthetic=True,
        metadata={
            "description": (
                "Deterministic synthetic dataset generated via scikit-learn make_classification. "
                "Feature names simulate spectral power in distinct EEG bands. "
                "NOTICE: This is synthetic data for academic testing and pipeline verification, "
                "NOT real physiological EEG recordings."
            ),
            "random_state": random_state,
            "test_size": test_size,
        }
    )


# Alias
generate_synthetic_eeg_dataset = get_default_sample_bundle
