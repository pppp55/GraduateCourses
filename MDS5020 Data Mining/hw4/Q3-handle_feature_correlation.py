# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:36:37 2025

@author: Neal
Review the provided code for detecting and managing correlated features,
focusing on the two methods: feature removal and PCA. And then, you need to

    - Analyze Step-1 and identify the code issue that results in future data "peeking"
        during the evaluation phase using train_test_split.

    - Correct the identified issue in Step-1 while ensuring all parameters,
        including threshold, n_components, test_size, and random seed,
        remain unchanged.

    - Run the code after resolving the code issue related to future data "peeking",
        then analyze the results to answer the questions in answer book

Note: Only modify the code in Step-1.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Generate a synthetic dataset with correlated features
# Binary classification, 5000 samples, 10 features (some will be correlated)
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=5,
    n_redundant=3,  # Introduce redundancy (correlated features)
    weights=[0.9, 0.1],
    random_state=42,
)  # Imbalanced for realism

# Convert to DataFrame for easier handling
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)

# fix random seed
random_seed = 12

# Manually add high correlation between some features (for demo)
np.random.seed(random_seed)
df["feature_8"] = df["feature_0"] * 0.95 + np.random.normal(
    0, 0.1, len(df)
)  # Highly correlated with feature_0
df["feature_9"] = df["feature_1"] * 0.85 + np.random.normal(
    0, 0.1, len(df)
)  # Highly correlated with feature_1

# %%  Step 1: Transform the training data if necessary and split it to train and holdout test
# ++Rewrite the code in Step 1++
X_train, X_test, y_train, y_test = train_test_split(
    df,
    y,
    test_size=0.2,
    stratify=y,
    random_state=random_seed,
)

# Make explicit copies so subsequent transformations never touch the holdout data inadvertently
X_train = X_train.copy()
X_test = X_test.copy()


# Method 1: Remove correlated features (threshold=0.8) using training data only
def remove_highly_correlated(df_in, threshold):
    corr_matrix = df_in.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Features to drop: {to_drop}")
    return to_drop


features_to_drop = remove_highly_correlated(X_train, threshold=0.8)
X_train_red = X_train.drop(columns=features_to_drop)
X_test_red = X_test.drop(columns=features_to_drop)
print(
    f"Original features: {X_train.shape[1]}, Reduced features: {X_train_red.shape[1]}"
)

# Method 2: Apply PCA (retain components explaining 95% variance) fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(
    f"PCA reduced to {X_train_pca.shape[1]} components (explaining {sum(pca.explained_variance_ratio_):.2%} variance)"
)


# %%  Step 2: Train and evaluate simple model (GaussianNB) with holdout evaluation
model = GaussianNB()

# Original (with correlations)
model.fit(X_train, y_train)
y_pred_orig = model.predict(X_test)
f1_orig = f1_score(y_test, y_pred_orig)

# After removal
model.fit(X_train_red, y_train)
y_pred_red = model.predict(X_test_red)
f1_red = f1_score(y_test, y_pred_red)

# After PCA
model.fit(X_train_pca, y_train)
y_pred_pca = model.predict(X_test_pca)
f1_pcs = f1_score(y_test, y_pred_pca)

print("\nModel Performance (F1-Score):")
print(f"Original (with correlations): {f1_orig:.2f}")
print(f"After Removing Correlated Features: {f1_red:.2f}")
print(f"After PCA Reduction: {f1_pcs:.2f}")
