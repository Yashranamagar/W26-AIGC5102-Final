"""
k-Nearest Neighbours (k-NN) - MNIST Classification

This script trains a k-NN classifier on the PCA-reduced MNIST dataset
(95% variance retained) and performs systematic hyperparameter tuning
via Grid Search using an 80/20 stratified train/validation split.

The primary tuning objective is Macro F1-score. The script also reports
Macro Precision and Macro Recall.

Pipeline:
    1. Mount Google Drive (Colab).
    2. Load PCA-reduced train and test CSVs.
    3. Stratified 80/20 train/validation split (random_state=0).
    4. Grid Search using PredefinedSplit so tuning happens on the
       validation set rather than a generic k-fold CV.
    5. Refit best params on train-only, evaluate on validation set.
    6. Refit best params on full training set (train+val), evaluate on test.
    7. Save best hyperparameters, classification reports, and confusion
       matrix to the k-NN output folder.

Why PCA data:
    - k-NN depends on Euclidean distance calculations in feature space.
      In 784 dimensions, distance becomes less meaningful (curse of
      dimensionality) and prediction time grows with feature count.
      Using the 95% variance PCA data (~150 components) dramatically
      speeds up prediction while improving distance semantics.


Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python "/content/drive/MyDrive/AI Integration & Governance/Into to ML/Final Project/Model Training/k-NN/train_knn.py"
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# If running in Google Colab, mount Drive first:
#   from google.colab import drive
#   drive.mount('/content/drive')
BASE_DIR = "/content/drive/MyDrive/AI Integration & Governance/Into to ML/Final Project"

# Input: PCA-reduced (95% variance) training and test data
PCA_DIR = f"{BASE_DIR}/Preprocessing/PCA File Output"
TRAIN_FILE = f"{PCA_DIR}/mnist_train_pca_95variance.csv"
TEST_FILE = f"{PCA_DIR}/mnist_test_pca_95variance.csv"

# Output: folder for k-NN artifacts
OUTPUT_DIR = f"{BASE_DIR}/Model Training/k-NN"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_PARAMS_FILE = f"{OUTPUT_DIR}/knn_best_params.json"
VAL_REPORT_FILE = f"{OUTPUT_DIR}/knn_validation_report.txt"
TEST_REPORT_FILE = f"{OUTPUT_DIR}/knn_test_report.txt"
CONFUSION_MATRIX_FILE = f"{OUTPUT_DIR}/knn_test_confusion_matrix.png"
GRID_RESULTS_FILE = f"{OUTPUT_DIR}/knn_grid_search_results.csv"
SUMMARY_FILE = f"{OUTPUT_DIR}/knn_summary.txt"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_csv(path):
    """Load a PCA-reduced MNIST CSV and return features and labels."""
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].astype(int).values
    features = df.iloc[:, 1:].astype(np.float32).values
    print(f"  -> shape: {df.shape}")
    return features, labels


X_train_full, y_train_full = load_csv(TRAIN_FILE)
X_test, y_test = load_csv(TEST_FILE)

print(f"\nTraining set: {X_train_full.shape}, labels: {y_train_full.shape}")
print(f"Test set:     {X_test.shape}, labels: {y_test.shape}")


# ---------------------------------------------------------------------------
# Stratified 80/20 train/validation split
# ---------------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=RANDOM_STATE,
)
print(f"\nAfter 80/20 stratified split:")
print(f"  Train:      {X_tr.shape}")
print(f"  Validation: {X_val.shape}")


# ---------------------------------------------------------------------------
# Grid Search (tuning on the validation set via PredefinedSplit)
# ---------------------------------------------------------------------------
X_combined = np.vstack([X_tr, X_val])
y_combined = np.concatenate([y_tr, y_val])
test_fold = np.array([-1] * len(X_tr) + [0] * len(X_val))
ps = PredefinedSplit(test_fold=test_fold)

scoring = {
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}

# Hyperparameter search space for k-NN.
# - n_neighbors: the k value. Smaller k = lower bias, higher variance.
# - weights: 'uniform' treats all neighbors equally; 'distance' weights
#   closer neighbors more heavily (often wins on MNIST).
# - p: power parameter for Minkowski distance (1 = Manhattan, 2 = Euclidean).
param_grid = {
    "n_neighbors": [3, 5, 7],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}

base_model = KNeighborsClassifier(
    algorithm="auto",
    n_jobs=-1,          # Parallelize distance calculations
)

print("\nStarting GridSearchCV ...")
print(f"  param_grid: {param_grid}")
print(f"  scoring:    {list(scoring.keys())}")
print(f"  refit:      f1_macro (primary tuning objective)")

start = time.time()
grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=ps,
    scoring=scoring,
    refit=False,        # We'll retrain the best model manually for clarity
    n_jobs=-1,
    verbose=2,
)
grid.fit(X_combined, y_combined)
elapsed = time.time() - start
print(f"\nGrid search complete in {elapsed/60:.2f} minutes.")


# ---------------------------------------------------------------------------
# Extract best hyperparameters (optimized for macro F1)
# ---------------------------------------------------------------------------
cv_results = pd.DataFrame(grid.cv_results_)
best_idx = cv_results["mean_test_f1_macro"].idxmax()
best_params = cv_results.loc[best_idx, "params"]
best_val_f1 = cv_results.loc[best_idx, "mean_test_f1_macro"]
best_val_precision = cv_results.loc[best_idx, "mean_test_precision_macro"]
best_val_recall = cv_results.loc[best_idx, "mean_test_recall_macro"]

print("\n" + "=" * 60)
print("BEST HYPERPARAMETERS (optimized for Macro F1)")
print("=" * 60)
print(json.dumps(best_params, indent=2, default=str))
print(f"\nValidation Macro Precision: {best_val_precision:.6f}")
print(f"Validation Macro Recall:    {best_val_recall:.6f}")
print(f"Validation Macro F1:        {best_val_f1:.6f}")

with open(BEST_PARAMS_FILE, "w") as f:
    json.dump(
        {
            "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v))
                            for k, v in best_params.items()},
            "validation_macro_precision": float(best_val_precision),
            "validation_macro_recall": float(best_val_recall),
            "validation_macro_f1": float(best_val_f1),
            "grid_search_time_minutes": elapsed / 60,
        },
        f,
        indent=2,
    )
cv_results.to_csv(GRID_RESULTS_FILE, index=False)


# ---------------------------------------------------------------------------
# Evaluate on validation set (train on X_tr only)
# ---------------------------------------------------------------------------
print("\nTraining best model on X_tr only for validation metrics ...")
val_model = KNeighborsClassifier(
    algorithm="auto",
    n_jobs=-1,
    **best_params,
)
val_model.fit(X_tr, y_tr)
y_val_pred = val_model.predict(X_val)

val_precision = precision_score(y_val, y_val_pred, average="macro")
val_recall = recall_score(y_val, y_val_pred, average="macro")
val_f1 = f1_score(y_val, y_val_pred, average="macro")
val_accuracy = accuracy_score(y_val, y_val_pred)
val_classification = classification_report(y_val, y_val_pred, digits=4)

print(f"\nValidation Macro Precision: {val_precision:.6f}")
print(f"Validation Macro Recall:    {val_recall:.6f}")
print(f"Validation Macro F1:        {val_f1:.6f}")
print(f"Validation Accuracy:        {val_accuracy:.6f}")

with open(VAL_REPORT_FILE, "w") as f:
    f.write("k-NN - Validation Set Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Best params: {best_params}\n\n")
    f.write(f"Macro Precision: {val_precision:.6f}\n")
    f.write(f"Macro Recall:    {val_recall:.6f}\n")
    f.write(f"Macro F1:        {val_f1:.6f}\n")
    f.write(f"Accuracy:        {val_accuracy:.6f}\n\n")
    f.write("Per-class classification report:\n")
    f.write(val_classification)


# ---------------------------------------------------------------------------
# Evaluate on test set (train on X_train_full = X_tr + X_val)
# ---------------------------------------------------------------------------
print("\nTraining final model on full training set for test metrics ...")
final_model = KNeighborsClassifier(
    algorithm="auto",
    n_jobs=-1,
    **best_params,
)
final_model.fit(X_train_full, y_train_full)
y_test_pred = final_model.predict(X_test)

test_precision = precision_score(y_test, y_test_pred, average="macro")
test_recall = recall_score(y_test, y_test_pred, average="macro")
test_f1 = f1_score(y_test, y_test_pred, average="macro")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification = classification_report(y_test, y_test_pred, digits=4)

print(f"\nTest Macro Precision: {test_precision:.6f}")
print(f"Test Macro Recall:    {test_recall:.6f}")
print(f"Test Macro F1:        {test_f1:.6f}")
print(f"Test Accuracy:        {test_accuracy:.6f}")

with open(TEST_REPORT_FILE, "w") as f:
    f.write("k-NN - Test Set Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Best params: {best_params}\n\n")
    f.write(f"Macro Precision: {test_precision:.6f}\n")
    f.write(f"Macro Recall:    {test_recall:.6f}\n")
    f.write(f"Macro F1:        {test_f1:.6f}\n")
    f.write(f"Accuracy:        {test_accuracy:.6f}\n\n")
    f.write("Per-class classification report:\n")
    f.write(test_classification)


# ---------------------------------------------------------------------------
# Confusion matrix (test set)
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
ax.set_title("k-NN - Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
plt.close(fig)
print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_FILE}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
summary = [
    "k-NN Training Summary",
    "=" * 60,
    f"Input data:              PCA 95% variance ({X_train_full.shape[1]} features)",
    f"Training samples:        {X_tr.shape[0]}",
    f"Validation samples:      {X_val.shape[0]}",
    f"Test samples:            {X_test.shape[0]}",
    f"Grid search time:        {elapsed/60:.2f} minutes",
    "",
    f"BEST HYPERPARAMETERS (optimized for Macro F1):",
    f"  {best_params}",
    "",
    f"Validation Macro Precision: {val_precision:.6f}",
    f"Validation Macro Recall:    {val_recall:.6f}",
    f"Validation Macro F1:        {val_f1:.6f}",
    f"Validation Accuracy:        {val_accuracy:.6f}",
    "",
    f"Test Macro Precision:       {test_precision:.6f}",
    f"Test Macro Recall:          {test_recall:.6f}",
    f"Test Macro F1:              {test_f1:.6f}",
    f"Test Accuracy:              {test_accuracy:.6f}",
]

summary_text = "\n".join(summary)
print("\n" + summary_text)

with open(SUMMARY_FILE, "w") as f:
    f.write(summary_text)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Done.")
