"""
PCA Dimensionality Reduction for MNIST

This script loads the MinMax-normalized MNIST datasets and applies Principal
Component Analysis (PCA) to reduce the 784 pixel features down to a smaller
set of components that retain ~95% of the original variance. This dramatically
speeds up training for distance-based models (SVM, k-NN) with minimal loss
in classification accuracy.

"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# If running in Google Colab, mount Drive first with:
#   from google.colab import drive
#   drive.mount('/content/drive')
BASE_DIR = "/content/drive/MyDrive/AI Integration & Governance/Into to ML/Final Project"
PREPROCESSING_DIR = f"{BASE_DIR}/Preprocessing"
OUTPUT_DIR = f"{PREPROCESSING_DIR}/PCA File Output"

TRAIN_FILE = f"{PREPROCESSING_DIR}/mnist_train_normalized.csv"
TEST_FILE = f"{PREPROCESSING_DIR}/mnist_test_normalized.csv"

TRAIN_OUTPUT_95 = f"{OUTPUT_DIR}/mnist_train_pca_95variance.csv"
TEST_OUTPUT_95 = f"{OUTPUT_DIR}/mnist_test_pca_95variance.csv"

TRAIN_OUTPUT_50 = f"{OUTPUT_DIR}/mnist_train_pca_50components.csv"
TEST_OUTPUT_50 = f"{OUTPUT_DIR}/mnist_test_pca_50components.csv"

TRAIN_OUTPUT_100 = f"{OUTPUT_DIR}/mnist_train_pca_100components.csv"
TEST_OUTPUT_100 = f"{OUTPUT_DIR}/mnist_test_pca_100components.csv"

SUMMARY_FILE = f"{OUTPUT_DIR}/pca_summary.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_mnist_csv(path):
    """Load a normalized MNIST CSV file and split into labels/features."""
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].astype(int)
    features = df.iloc[:, 1:].astype(np.float32)
    print(f"  -> shape: {df.shape} (labels: {labels.shape}, features: {features.shape})")
    return labels, features


def apply_pca(train_features, test_features, n_components, label):
    """Fit PCA on training features and transform both train and test."""
    print(f"\nFitting PCA ({label}) ...")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    train_reduced = pca.fit_transform(train_features)
    test_reduced = pca.transform(test_features)
    n_final = pca.n_components_
    variance_kept = pca.explained_variance_ratio_.sum()
    print(f"  -> components selected: {n_final}")
    print(f"  -> total variance explained: {variance_kept:.4f}")
    return pca, train_reduced, test_reduced, n_final, variance_kept


def save_reduced(labels, reduced_features, output_path):
    """Save reduced features and labels back to a CSV."""
    n_components = reduced_features.shape[1]
    column_names = [f"pc{i+1}" for i in range(n_components)]
    df = pd.DataFrame(reduced_features, columns=column_names)
    df.insert(0, "label", labels.values)
    df.to_csv(output_path, index=False)
    print(f"  -> saved: {output_path}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train_labels, train_features = load_mnist_csv(TRAIN_FILE)
test_labels, test_features = load_mnist_csv(TEST_FILE)

print("\nSanity checks BEFORE PCA:")
print(f"  Train feature range: [{train_features.values.min():.4f}, {train_features.values.max():.4f}]")
print(f"  Test  feature range: [{test_features.values.min():.4f}, {test_features.values.max():.4f}]")


# ---------------------------------------------------------------------------
# Ensure output directory exists
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Variant 1: Retain 95% of variance (automatic component selection)
# This is the main experimental branch recommended for the report.
# ---------------------------------------------------------------------------
pca_95, train_95, test_95, n_95, var_95 = apply_pca(
    train_features, test_features, n_components=0.95, label="95% variance"
)
print("Saving 95% variance PCA datasets ...")
save_reduced(train_labels, train_95, TRAIN_OUTPUT_95)
save_reduced(test_labels, test_95, TEST_OUTPUT_95)


# ---------------------------------------------------------------------------
# Variant 2: Fixed 50 components
# ---------------------------------------------------------------------------
pca_50, train_50, test_50, n_50, var_50 = apply_pca(
    train_features, test_features, n_components=50, label="50 components"
)
print("Saving 50-component PCA datasets ...")
save_reduced(train_labels, train_50, TRAIN_OUTPUT_50)
save_reduced(test_labels, test_50, TEST_OUTPUT_50)


# ---------------------------------------------------------------------------
# Variant 3: Fixed 100 components
# ---------------------------------------------------------------------------
pca_100, train_100, test_100, n_100, var_100 = apply_pca(
    train_features, test_features, n_components=100, label="100 components"
)
print("Saving 100-component PCA datasets ...")
save_reduced(train_labels, train_100, TRAIN_OUTPUT_100)
save_reduced(test_labels, test_100, TEST_OUTPUT_100)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
summary_lines = [
    "PCA Dimensionality Reduction Summary",
    "=" * 45,
    f"Original feature count:            {train_features.shape[1]}",
    f"Training samples:                  {train_features.shape[0]}",
    f"Test samples:                      {test_features.shape[0]}",
    "",
    "Variant 1: 95% variance retained",
    f"  components selected:             {n_95}",
    f"  variance explained:              {var_95:.6f}",
    "",
    "Variant 2: 50 fixed components",
    f"  components selected:             {n_50}",
    f"  variance explained:              {var_50:.6f}",
    "",
    "Variant 3: 100 fixed components",
    f"  components selected:             {n_100}",
    f"  variance explained:              {var_100:.6f}",
    "",
    "Notes:",
    "  - PCA was fit on the training set only and applied to both",
    "    train and test sets to prevent data leakage.",
    "  - random_state=0 was used per the project rubric.",
    "  - Use the 95% variance dataset as the primary PCA branch;",
    "    use the 50/100 component variants for ablation comparison",
    "    in the report's preprocessing discussion.",
]

with open(SUMMARY_FILE, "w") as f:
    f.write("\n".join(summary_lines))

print("\n" + "\n".join(summary_lines))
print(f"\nSummary saved to: {SUMMARY_FILE}")
print("\nDone. PCA dimensionality reduction complete.")
