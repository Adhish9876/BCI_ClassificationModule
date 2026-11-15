# 📄 Project Contribution Statement: BCI Classification Lab (Module 8)

**Course**: Brain-Computer Interfaces (BCI) Laboratory Project  
**Module**: Module 8 — Feature Classification, Diagnostics & Benchmark Lab  
**Total Team Weight**: 100% (Equally distributed at 25.0% per member)  
**Date**: November 2026  

---

## 👥 Equal Workload Distribution Matrix (25% Each)

The responsibilities and deliverables for this project were planned, implemented, and verified in strict accordance with the equal-contribution mandate. Each member spearheaded a core domain of the engineering lifecycle:

| Team Member | Functional Role | Primary Focus Area | Workload Contribution |
|:---|:---|:---|:---:|
| **Member A** | **Machine Learning Lead** | Scikit-learn Pipeline Architecture, Cross-Validation & Validation Engine | **25%** |
| **Member B** | **Scientific UI / Frontend Lead** | Streamlit Interactive Laboratory, Stepper Layout & Custom Dark/Light Theming | **25%** |
| **Member C** | **Diagnostics & Visualization Lead** | Linked Reactive Plots, Confusion Matrices, Metric Engines & Educational Q&A | **25%** |
| **Member D** | **Quality Assurance & DevOps Lead** | Automated Test Suite (21 Pytests), Serialization, Artifact Exports & Documentation | **25%** |
| **Total** | | | **100.0%** |

---

## 🛠️ Detailed Member Responsibilities & Technical Contributions

### 1. Member A — Machine Learning & Pipeline Engineering (25%)
- **Architecture**: Designed the core Python classification engine (`classification/schemas.py`, `classification/preprocessing.py`, `classification/model_registry.py`).
- **Pipeline Implementation**: Constructed scikit-learn `Pipeline` objects bundling configurable transformers (`StandardScaler`, `RobustScaler`, `MinMaxScaler`) with classifiers (`LinearDiscriminantAnalysis`, `SVC`, `RandomForestClassifier`, `KNeighborsClassifier`, `LogisticRegression`).
- **Validation Rigor**: Implemented stratified $k$-fold cross-validation and participant-aware `GroupKFold` splitting to mathematically prevent subject identity leakage.
- **Synthetic BCI Generator**: Developed realistic synthetic EEG datasets (`classification/sample_data.py`) simulating Motor Imagery (mu/beta ERD/ERS), P300 ERPs (imbalanced), and Alpha Neurofeedback.

### 2. Member B — UI/UX & Frontend Architecture (25%)
- **Application Shell**: Architected the 3-step breadcrumb stepper (`ui/layout.py`) guiding users through Data Ingestion, Model Configuration, and Benchmark Telemetry.
- **Data Ingestion View**: Built Step 1 UI (`ui/data_page.py`) with dual-mode support for built-in BCI paradigm switching and flexible multi-file CSV uploads.
- **Training Workbench**: Built Step 2 UI (`ui/model_page.py`) featuring model selector toggles, hyperparameter popovers, and interactive validation parameter sliders.
- **Styling & Theming**: Formatted high-contrast, professional scientific aesthetics (`ui/styles.py`), responsive containers, and dark-mode code editors.

### 3. Member C — Interactive Diagnostics & Educational Q&A (25%)
- **Linked Reactive Plots**: Built dual linked Plotly visualizations (`ui/components.py`, `ui/results_page.py`) synchronizing confusion matrices with dynamic model-inspection charts.
- **Diagnostic Metrics**: Implemented multi-metric calculation (`classification/evaluation.py`), including Accuracy, Balanced Accuracy, Macro Precision/Recall/F1, and Cohen's Kappa ($\kappa$).
- **Educational Explain Panel**: Authored technically rigorous Q&A modules contextualizing BCI classification nuances (e.g., majority-class accuracy traps, Gaussian assumptions in LDA, and Gini feature importance).
- **Telemetry Dashboards**: Designed executive summary KPI banner and sortable comparative leaderboard tables.

### 4. Member D — DevOps, Testing & Artifact Export Manager (25%)
- **Automated Test Suite**: Authored and executed 21 comprehensive unit tests (`tests/test_data_validation.py`, `test_model_registry.py`, `test_training.py`, `test_evaluation.py`, `test_exports.py`) verifying pipeline execution, metrics calculation, and zero data leakage.
- **Artifact Export Engine**: Developed standalone Python script generator (`classification/export.py`), binary `.joblib` model serializer, benchmark `.csv` exporter, and `.json` parameter manifests.
- **Artifact Manager UI**: Implemented segmented tabbed workspace in Step 3 for executable assets and scientific reports with code preview and copy actions.
- **Documentation**: Structured `README.md`, developer integration guidelines, and `USER_GUIDE.md`.

---

## ⏱️ 10-Minute Presentation & Live Demo Agenda (All Members Speak)

To ensure equal representation during oral examination and project demo, each team member is allocated exactly 2.5 minutes:

```
[00:00 - 02:30] Member A ──> [02:30 - 05:00] Member B ──> [05:00 - 07:30] Member C ──> [07:30 - 10:00] Member D
```

| Time Slot | Speaker | Demo Focus & Live Actions |
|:---|:---|:---|
| **00:00 – 02:30** | **Member A** | **Introduction & Pipeline Rigor**: Introduce BCI classification motivation; demonstrate dataset ingestion, 18-channel EEG feature representations, and subject-aware group splitting. |
| **02:30 – 05:00** | **Member B** | **Live Training & Parameter Control**: Walk through Step 2 workbench; adjust feature scaling (RobustScaler), toggle LDA/SVM/RF models, tune hyperparameters in real-time, and trigger cross-validation. |
| **05:00 – 07:30** | **Member C** | **Benchmarking & Linked Diagnostics**: Analyze Step 3 leaderboard rankings; demonstrate linked confusion matrix updates and RF feature importances; walk through the educational Explain Panel. |
| **07:30 – 10:00** | **Member D** | **Testing & Reproducible Artifacts**: Run automated pytest test suite (21 passing tests); demonstrate code preview, download fitted `.joblib` pipeline and show instant inference verification in terminal. |

---

## ✍️ Verification & Signatures

All members affirm that this statement accurately reflects equal effort, collaborative peer review, and shared intellectual contribution to the repository.

* **Member A**: ___________________________ &nbsp;&nbsp;&nbsp;&nbsp; Date: ______________
* **Member B**: ___________________________ &nbsp;&nbsp;&nbsp;&nbsp; Date: ______________
* **Member C**: ___________________________ &nbsp;&nbsp;&nbsp;&nbsp; Date: ______________
* **Member D**: ___________________________ &nbsp;&nbsp;&nbsp;&nbsp; Date: ______________
