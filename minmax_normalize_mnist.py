"""
MinMax Normalization for MNIST CSV Datasets
AIGC 5102 - Final Project: Handwritten Digit Classification

This script loads mnist_train.csv and mnist_test.csv, applies MinMax
normalization to the pixel features (scaling them from [0, 255] to [0, 1]),
and saves the normalized datasets as new CSV files.

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# Paths
# If running in Google Colab, mount Drive first with:
#   from google.colab import drive
#   drive.mount('/content/drive')
BASE_DIR = "/content/drive/MyDrive/AI Integration & Governance/Into to ML/Final Project"
DATASET_DIR = f"{BASE_DIR}/Dataset"
OUTPUT_DIR = f"{BASE_DIR}/Preprocessing"

TRAIN_FILE = f"{DATASET_DIR}/mnist_train.csv"
TEST_FILE = f"{DATASET_DIR}/mnist_test.csv"

TRAIN_OUTPUT = f"{OUTPUT_DIR}/mnist_train_normalized.csv"
TEST_OUTPUT = f"{OUTPUT_DIR}/mnist_test_normalized.csv"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_mnist_csv(path):
    """Load an MNIST CSV file and split into features and labels."""
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].astype(int)
    features = df.iloc[:, 1:].astype(np.float32)
    print(f"  -> shape: {df.shape} (labels: {labels.shape}, features: {features.shape})")
    return labels, features


train_labels, train_features = load_mnist_csv(TRAIN_FILE)
test_labels, test_features = load_mnist_csv(TEST_FILE)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
print("\nSanity checks BEFORE normalization:")
print(f"  Train pixel range: [{train_features.values.min()}, {train_features.values.max()}]")
print(f"  Test  pixel range: [{test_features.values.min()}, {test_features.values.max()}]")
print(f"  Any NaNs in train? {train_features.isna().any().any()}")
print(f"  Any NaNs in test?  {test_features.isna().any().any()}")


# ---------------------------------------------------------------------------
# MinMax normalization
# ---------------------------------------------------------------------------
# Fit the scaler ONLY on the training data to prevent data leakage,
# then transform both training and test sets with the same scaler.
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

# Note: For MNIST, pixel values are bounded in [0, 255] so this is
# mathematically equivalent to dividing by 255.0, but using MinMaxScaler
# keeps the preprocessing reusable and explicit.


# ---------------------------------------------------------------------------
# Reassemble labeled DataFrames
# ---------------------------------------------------------------------------
feature_columns = train_features.columns

train_normalized_df = pd.DataFrame(train_scaled, columns=feature_columns)
train_normalized_df.insert(0, "label", train_labels.values)

test_normalized_df = pd.DataFrame(test_scaled, columns=feature_columns)
test_normalized_df.insert(0, "label", test_labels.values)


# ---------------------------------------------------------------------------
# Save normalized datasets
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving normalized datasets to: {OUTPUT_DIR}")
train_normalized_df.to_csv(TRAIN_OUTPUT, index=False)
test_normalized_df.to_csv(TEST_OUTPUT, index=False)
print(f"  -> {TRAIN_OUTPUT}")
print(f"  -> {TEST_OUTPUT}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
print("\nSanity checks AFTER normalization:")
print(f"  Train pixel range: [{train_scaled.min():.4f}, {train_scaled.max():.4f}]")
print(f"  Test  pixel range: [{test_scaled.min():.4f}, {test_scaled.max():.4f}]")
print(f"  Train mean: {train_scaled.mean():.4f} | std: {train_scaled.std():.4f}")
print(f"  Test  mean: {test_scaled.mean():.4f} | std: {test_scaled.std():.4f}")

print("\nDone. MinMax normalization complete.")
