# Build an EEG Classification Lab – BCI Course Project

Act as a senior Python developer, machine learning engineer, and UI/UX designer. I want you to build a complete, functional, professionally designed **EEG Classification Lab** as part of a larger Brain-Computer Interface (BCI) teaching application.

This is an academic engineering project. The application should teach students how different conventional machine learning classifiers work on EEG-derived features, how models are trained, and how their performance can be compared.

Do not create a basic demonstration dashboard. Build a genuinely interactive application with actual machine learning functionality, proper data handling, meaningful visualizations, and a polished user interface.

---

# 1. PROJECT CONTEXT

This is one module in a larger BCI teaching application.

Different student teams are developing the following modules:

1. Data Loading
2. Filtering
3. Artifact Handling
4. Epoching and ERP
5. Time-Frequency Analysis
6. Spatial Analysis
7. Feature Extraction
8. Classification (OUR MODULE)
9. Paradigm Explorer

Eventually, all modules will be integrated into one Streamlit application.

Our module is responsible exclusively for classification.

The Feature Extraction module will produce numerical feature matrices and their corresponding class labels. Our module will receive those features and labels, train selected machine learning classifiers, evaluate them, and compare their performance.

**Important:** We do not implement feature extraction, filtering, artifact removal, or other upstream EEG processing operations. Those are handled by other teams.

However, our module must be independently functional during development because the other teams may not have completed their implementations.

---

# 2. CORE OBJECTIVE

Build an interactive comparative machine learning environment where users can:

1. Load an EEG feature dataset.
2. Supply separate training and testing datasets, or create a train-test split.
3. Inspect and validate the data.
4. Select multiple machine learning classifiers.
5. Configure each classifier's hyperparameters.
6. Train all selected classifiers by clicking a single button.
7. Evaluate their predictions against unseen test data.
8. Compare their performance using metrics and visualizations.
9. Inspect individual classifier results.
10. Export experiment results and configurations.

The application must perform actual model training using scikit-learn.

Do not use hardcoded accuracy values, precomputed results, or simulated training.

---

# 3. CLASSIFIERS TO IMPLEMENT

Implement the following five classifiers as the primary algorithms.

## A. Linear Discriminant Analysis (LDA)

Library: `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`

Explain:
- What LDA does.
- How it separates classes.
- Why it is commonly used as a baseline in BCI classification.

Hyperparameters:
- Solver
- Shrinkage

Provide valid parameter combinations. For example, shrinkage must not be enabled with the SVD solver.

## B. Support Vector Machine (SVM)

Library: `sklearn.svm.SVC`

Explain:
- Support vectors.
- Maximum-margin classification.
- Linear versus nonlinear kernels.

Hyperparameters:
- Kernel: Linear, RBF, Polynomial.
- C.
- Gamma.
- Degree (for polynomial kernel).

Use feature scaling inside a scikit-learn Pipeline.

## C. Random Forest

Library: `sklearn.ensemble.RandomForestClassifier`

Explain:
- Decision trees.
- Ensemble learning.
- Majority voting.

Hyperparameters:
- Number of estimators.
- Maximum depth.
- Minimum samples split.
- Class weighting.

Include feature importance visualization where appropriate.

## D. K-Nearest Neighbors (KNN)

Library: `sklearn.neighbors.KNeighborsClassifier`

Explain:
- Distance-based classification.
- Nearest neighbours.
- How changing K affects predictions.

Hyperparameters:
- Number of neighbours.
- Distance metric.
- Weighting method.

Use feature scaling inside a Pipeline.

## E. Logistic Regression

Library: `sklearn.linear_model.LogisticRegression`

Explain:
- Logistic function.
- Probability-based classification.
- Regularization.

Hyperparameters:
- Regularization strength C.
- Solver.
- Penalty.
- Class weighting.

Use valid solver and penalty combinations.

### Optional BCI-specific extension

Consider adding Riemannian Minimum Distance to Mean (MDM) using PyRiemann as an optional sixth classifier.

Only enable it when the input is actually a suitable covariance-matrix representation.

Do not apply MDM to an ordinary flat feature matrix.

If PyRiemann is unavailable, the other five classifiers must continue working normally.

**Do not implement CNN, EEGNet, or deep learning models.**

---

# 4. DATA INPUT AND HANDLING

This is extremely important.

Our module must work independently without depending on the other teams.

Implement three data input methods.

## Method 1: Built-in sample dataset

Create a deterministic sample dataset using scikit-learn's `make_classification`.

Suggested configuration:
- 600–1000 samples.
- 12–24 features.
- Two or three classes.
- Fixed random seed.

Clearly identify this dataset as synthetic demonstration data.

Do not represent synthetic data as actual EEG recordings.

This dataset is primarily for testing the application's training and evaluation workflow.

## Method 2: Upload custom datasets

Allow users to upload:

- X_train.csv
- y_train.csv
- X_test.csv
- y_test.csv

Expected format:

- Rows represent samples.
- Columns represent extracted features.
- Labels represent the corresponding class.

Also support uploading one labelled CSV where the user can select the target label column and configure a train-test split.

Validate:
- Matching training feature and label counts.
- Matching testing feature and label counts.
- Consistent feature columns.
- Numeric feature values.
- Missing and infinite values.
- Valid class labels.
- At least two classes in training data.

Do not silently discard invalid data.

Provide clear error messages.

## Method 3: Integration with Feature Extraction

Eventually, our module will receive data from the Feature Extraction team.

The expected interface is:

`X_train, y_train, X_test, y_test`

The classification module must be designed to accept this data directly.

Do not make the entire module dependent on the implementation of another team.

---

# 5. MODEL TRAINING WORKFLOW

This is the central functionality of our application.

The actual model training and evaluation must happen inside the Classification module.

The workflow should be:

**Step 1 – Receive data**

Receive:
- X_train
- y_train
- X_test
- y_test

**Step 2 – Select models**

Allow users to select one or multiple classifiers.

**Step 3 – Configure models**

Allow users to configure the hyperparameters of each selected classifier.

**Step 4 – Train and evaluate**

Provide one prominent button:

"TRAIN AND EVALUATE"

When clicked:

For each selected classifier:

1. Instantiate the classifier with the selected configuration.
2. Apply the appropriate preprocessing pipeline.
3. Fit the model using X_train and y_train.
4. Generate predictions using X_test.
5. Compare predictions against y_test.
6. Calculate evaluation metrics.
7. Record training and prediction time.
8. Store the trained estimator and experiment results.

All selected models must use the same training and testing data for a fair comparison.

If one classifier fails, do not discard the successful results of other classifiers.

Show meaningful progress and error messages.

**Step 5 – Display results**

Display a comparative results dashboard containing actual computed results.

---

# 6. EVALUATION METRICS

Implement the following metrics using scikit-learn.

1. Accuracy
2. Balanced Accuracy
3. Precision (Macro)
4. Precision (Weighted)
5. Recall (Macro)
6. Recall (Weighted)
7. F1-score (Macro)
8. F1-score (Weighted)
9. Cohen's Kappa
10. Confusion Matrix

Also provide:
- Per-class precision.
- Per-class recall.
- Per-class F1-score.
- Support for each class.
- Model training time.
- Model prediction time.

For imbalanced EEG datasets, emphasize macro F1 and balanced accuracy.

Do not rely exclusively on accuracy.

ROC-AUC can be added optionally when valid probability or decision scores are available.

---

# 7. TRAINING AND TESTING METHODOLOGY

Implement scientifically correct data handling.

If separate training and testing files are supplied, use them directly.

If a single labelled dataset is supplied, allow the user to configure a train-test split.

Default split:
- 80% training.
- 20% testing.
- Stratified splitting.
- Random seed: 42.

Also provide participant-aware splitting when participant IDs are available.

This is particularly important for EEG because multiple epochs from the same participant can be correlated.

Provide an option for group-based splitting so that samples belonging to the same participant do not appear in both training and testing when evaluating subject-independent generalization.

Prevent data leakage:

- Fit scalers only on training data.
- Fit imputers only on training data.
- Never use testing data during model fitting.
- Never use the test set for hyperparameter tuning.
- Use scikit-learn Pipelines.

Optional advanced functionality:
- Cross-validation.
- GridSearchCV.
- RandomizedSearchCV.

Cross-validation and hyperparameter tuning must operate on training data only. Keep the final test set untouched.

---

# 8. USER INTERFACE DESIGN

This is one of the most important aspects of the project.

I want a beautiful, professionally designed interface that looks like a genuine scientific/engineering application.

I do NOT want:
- Generic AI-generated dashboard styling.
- Excessive gradients.
- Glowing elements.
- Neon colours.
- Excessive glassmorphism.
- Random decorative graphics.
- Giant dashboard cards.
- Too many unnecessary icons.
- Overly rounded components.
- Artificial animations.
- Fake live indicators.
- Unnecessary visual complexity.

The interface should feel like a thoughtfully designed university research lab application.

## Design direction

Use a restrained, modern scientific aesthetic.

Suggested colour palette:

- Background: Warm off-white (#F5F3EE).
- Main text: Deep navy (#17243D).
- Primary accent: Muted teal (#147D78).
- Secondary text: Slate grey (#647084).
- Cards: White (#FFFFFF).
- Borders: Subtle warm grey (#DEDCD4).

Use clean typography, good spacing, subtle borders, and strong visual hierarchy.

Avoid making everything look like a card.

Use a consistent grid and alignment.

The interface should be visually polished without sacrificing functionality.

## Main application layout

### Header

Title:

**Classification Lab**

Subtitle:

"Train, evaluate and compare machine learning models on EEG-derived features."

Show a small indicator identifying the current dataset source.

### Navigation

Organize the workflow into three sections:

1. DATA
2. MODELS
3. RESULTS

Use simple tabs, a sidebar, or restrained step navigation.

Avoid excessive navigation elements.

---

# 9. DATA SECTION

Design a clean dataset configuration interface.

Include:

- Dataset source selection.
- Built-in sample dataset option.
- Custom file upload.
- Integrated pipeline data option.
- Training dataset information.
- Testing dataset information.

Display useful dataset statistics:

- Number of samples.
- Number of features.
- Number of classes.
- Class distribution.
- Feature names.

Include a small data preview table.

Use a simple chart to display class distribution.

Provide clear validation messages.

The user should immediately understand whether the dataset is ready for classification.

---

# 10. MODELS SECTION

This is where users select and configure classifiers.

Show the five classifiers as clean, selectable components.

Each classifier should include:

- Model name.
- Short description.
- A brief explanation of its working principle.
- Selection checkbox.
- Expandable hyperparameter settings.

For example:

**Support Vector Machine**

"Finds an optimal decision boundary that separates different classes."

Settings:
- Kernel.
- C.
- Gamma.
- Degree.

**Random Forest**

"Combines multiple decision trees to improve classification performance."

Settings:
- Number of trees.
- Maximum depth.
- Minimum samples split.

Allow users to select multiple classifiers simultaneously.

Clearly show how many classifiers are selected.

At the bottom, show:

- Selected dataset.
- Training sample count.
- Testing sample count.
- Selected classifiers.

Then provide the primary action:

**TRAIN AND EVALUATE**

This button must trigger actual model training.

Do not train models automatically whenever a checkbox or parameter changes.

---

# 11. RESULTS SECTION

After training, display a well-organized comparison interface.

## A. Performance overview

Display a compact summary of:

- Number of models successfully trained.
- Highest macro F1-score.
- Highest accuracy.
- Total experiment duration.

Only show actual results.

## B. Comparison table

Create a sortable comparison table:

| Model | Accuracy | Balanced Accuracy | Macro F1 | Kappa | Training Time |
|---|---|---|---|---|---|

Use actual computed metrics.

Highlight the best result only with respect to the selected metric.

Do not claim that a classifier is universally superior.

## C. Performance comparison chart

Use Plotly to create a clean comparison chart.

Allow users to compare:

- Accuracy.
- Balanced Accuracy.
- Macro F1.
- Precision.
- Recall.

The chart should be readable and professionally styled.

Avoid excessive colours and unnecessary chart decorations.

## D. Confusion matrices

Provide an individual confusion matrix for each trained classifier.

Allow users to select a classifier and inspect its matrix.

Use class labels on both axes.

Include clear axis descriptions:

- Actual class.
- Predicted class.

## E. Detailed model evaluation

For each classifier, provide:

- Model configuration.
- Evaluation metrics.
- Classification report.
- Confusion matrix.
- Training time.
- Prediction time.

For Random Forest, additionally display feature importance when appropriate.

---

# 12. EXPLAINABILITY AND EDUCATIONAL VALUE

Remember that this is a BCI teaching application, not just a machine learning benchmark.

Include a small contextual "Understand this result" or "Explain" panel.

Examples:

- What does accuracy mean?
- Why might SVM outperform LDA?
- What does a confusion matrix indicate?
- Why is macro F1 useful for imbalanced datasets?
- Why does feature scaling matter for KNN and SVM?
- Why can Random Forest handle nonlinear relationships?
- What does Cohen's Kappa measure?

The explanations should be concise, technically correct, and relevant to the current experiment.

Do not make unsupported claims about why a particular model performed better. Explain possible reasons rather than presenting speculation as fact.

---

# 13. EXPORT FUNCTIONALITY

Allow users to export:

- Model comparison results as CSV.
- Experiment configuration as JSON.
- Individual classification reports.
- Confusion matrices.
- Trained model files using joblib (optional).

Include experiment metadata:

- Dataset name.
- Dataset source.
- Number of samples.
- Number of features.
- Selected classifiers.
- Hyperparameters.
- Train-test split.
- Random seed.
- Evaluation metrics.
- Software versions.

Warn users that joblib/pickle model files should only be loaded from trusted sources.

---

# 14. TECHNICAL STACK

Use:

| Component | Technology |
|---|---|
| Application | Streamlit |
| Language | Python |
| Data handling | NumPy, pandas |
| Machine learning | scikit-learn |
| BCI-specific extension | PyRiemann |
| Visualization | Plotly |
| EEG extensions | MNE-Python (optional) |
| Testing | pytest |
| Model serialization | joblib |

Keep the dependencies minimal.

Do not install unnecessary libraries.

---

# 15. RECOMMENDED PROJECT STRUCTURE

Use a modular architecture.

```text
classification-module/
│
├── app.py
├── requirements.txt
├── README.md
│
├── classification/
│   ├── schemas.py
│   ├── data_loader.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── model_registry.py
│   ├── training.py
│   ├── evaluation.py
│   ├── export.py
│   └── sample_data.py
│
├── ui/
│   ├── layout.py
│   ├── components.py
│   ├── styles.py
│   ├── data_page.py
│   ├── model_page.py
│   └── results_page.py
│
└── tests/
    ├── test_data_validation.py
    ├── test_model_registry.py
    ├── test_training.py
    ├── test_evaluation.py
    └── test_exports.py
```

Keep the model logic independent of Streamlit.

The main training function should conceptually look like:

```python
def train_and_evaluate(
    dataset,
    selected_models,
    model_configs
):
    # Train selected models
    # Generate predictions
    # Calculate metrics
    # Return experiment results
```

This makes the classification engine independently testable and reusable by the main BCI application.

---

# 16. INTEGRATION WITH THE MAIN BCI APPLICATION

Eventually, the Classification module will be embedded into the shared Streamlit application.

The other teams will develop their modules independently.

Our module must expose a clean integration interface, for example:

```python
def render_classification_module(dataset=None):
    ...
```

When the Feature Extraction module is ready, the parent application should pass its output to our module.

Expected data:

```python
X_train
y_train

X_test
y_test
```

Our module then takes responsibility for all classification operations.

The upstream team must NOT train classifiers.

The Classification team must NOT repeat feature extraction.

The data contract must be documented clearly so that integration does not require rewriting the training engine.

For standalone development, the module should use the built-in dataset or uploaded files.

Do not assume other teams' implementations exist.

---

# 17. TESTING REQUIREMENTS

Implement automated tests using pytest.

At minimum, test:

1. Dataset loading and validation.
2. Invalid feature dimensions.
3. Missing values.
4. Class label consistency.
5. Successful training of all five classifiers.
6. Prediction generation.
7. Metric calculation.
8. Confusion matrix generation.
9. Multi-model comparison.
10. Export functionality.

The module must be tested without requiring the other teams' modules.

Use deterministic data and fixed random seeds.

---

# 18. IMPLEMENTATION INSTRUCTIONS

Do not generate the entire project blindly in one step.

Follow this development sequence.

### Phase 1: Inspect and understand

First inspect the current working directory and repository.

Understand:
- Existing project structure.
- Installed Python version.
- Existing dependencies.
- Whether this is a standalone module or part of the shared repository.

Do not overwrite existing work.

### Phase 2: Implement the classification engine

Implement:
- Data schema.
- Sample dataset generation.
- Dataset loading.
- Validation.
- Model registry.
- Hyperparameter configurations.
- Training pipelines.
- Evaluation metrics.

Verify that all five classifiers actually train and generate predictions.

### Phase 3: Build the UI

Implement:
- Dataset configuration.
- Dataset inspection.
- Classifier selection.
- Hyperparameter controls.
- Train and evaluate action.
- Results dashboard.
- Confusion matrices.
- Performance comparison.

Apply the visual design requirements described above.

### Phase 4: Exports and documentation

Implement result exports and write a proper README.

### Phase 5: Testing and refinement

Run all automated tests.

Launch the Streamlit application.

Test the complete workflow manually.

Fix any bugs before considering the implementation complete.

---

# 19. FINAL ACCEPTANCE CRITERIA

The project is complete only when:

- The application launches successfully.
- It works independently of other BCI modules.
- It provides a built-in sample dataset.
- Users can upload their own datasets.
- Users can select multiple classifiers.
- Users can configure model parameters.
- Clicking Train and Evaluate actually trains the selected models.
- All selected models use the same training and testing data.
- Models generate predictions on unseen test samples.
- All evaluation metrics are calculated from actual predictions.
- Confusion matrices are generated.
- Model comparison visualizations work.
- Results can be exported.
- Data leakage is prevented through proper preprocessing pipelines.
- Participant-aware splitting is available when participant IDs are supplied.
- Automated tests pass.
- The UI looks professionally designed and is easy to understand.
- The module can later be integrated into the shared BCI application without rewriting the classification engine.

---

# 20. MOST IMPORTANT INSTRUCTIONS

Prioritize these things in order:

1. **Correct machine learning implementation.**
2. **Reliable data handling and validation.**
3. **A clean and reusable architecture.**
4. **Meaningful educational explanations.**
5. **A beautiful, professional UI.**

Do not sacrifice functionality for visual appearance.

Do not make the application unnecessarily sophisticated.

Do not add features simply to make the project look impressive.

I want a practical, polished, well-engineered application that looks like something a university research lab would genuinely use to teach EEG classification.

Start by inspecting the repository, then explain your implementation plan briefly before proceeding. Implement the project in phases, verify each phase, and do not claim completion until the complete workflow has been tested.

**Build the actual application, not just a prototype or a UI mockup.**