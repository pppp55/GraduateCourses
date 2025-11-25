# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:16:24 2025

@author: Neal LONG

You need to investigate how different preprocessing strategies affect KNN model
performance on the bank marketing dataset. Your goal is to systematically evaluate
KNN models under various conditions and draw meaningful conclusions.

In detail, you find that the numerical feature "pdays" contains a special value
999 that might disproportionately influence distance calculations in KNN. Then
you plan to evaluate the following four feature engineering settings:
    - Setting 1 - "With pdays + StandardScaler": Include "pdays" as the `df_train_with_pdays`,
        use StandardScaler for numerical features,
        and OneHotEncoder for categorical features.
    - Setting 2 - "With pdays + Normalizer": Include "pdays" as the `df_train_with_pdays`,
        use Normalizer for numerical features,
        and OneHotEncoder for categorical features.
    - Setting 3 - "Without pdays + StandardScaler": Exclude "pdays" as the `df_train_without_pdays`,
        use StandardScaler for numerical features,
        and OneHotEncoder for categorical features.
    - Setting 4 - "Without pdays + Normalizer": Exclude "pdays" as the `df_train_without_pdays`,
        use Normalizer for numerical features,
        and OneHotEncoder for categorical features.

For each setting, utilize GridSearchCV with 5-fold cross-validation as defined `cv` to determine
the optimal value of K from the range [100, 200, 300, 400, 500, 600] for the KNN model.
Record and analyze the average AUC-ROC score of the best model for each setting.

Note:
    1. Use the provided `df_train_with_pdays`, `df_train_without_pdays`, `y_true` and `cv`
    2. Use make_column_transformer and make_column_selector to build the preprocessing pipelines.
    3. Answer the questions based on your analysis.


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline


data_dir = r"./data/bank_marketing_train.csv"

print("\n")
print("#" * 30)
print("Load training data:")
df_train_raw = pd.read_csv(data_dir)
print("Raw traininig data", df_train_raw.shape)
print("\n")
print("#" * 30)
print("Data clean and feature engineering:")


df_train_raw = df_train_raw.replace(to_replace={"unknown": np.nan}).infer_objects(
    copy=False
)
df_train_raw = df_train_raw.dropna()

# %% Define the df_train_with_pdays and df_train_without_pdays
y_true = df_train_raw.pop("y")

df_train_with_pdays = df_train_raw

df_train_without_pdays = df_train_raw.drop("pdays", axis=1)

print(
    "Shape of df_train_with_pdays is {} and df_train_without_pdays is {}".format(
        df_train_with_pdays.shape, df_train_without_pdays.shape
    )
)

#  Define the 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# %%
# ++insert your code below++


def evaluate_setting(features_df, scaler_cls, setting_name):
    """Run GridSearchCV for a single preprocessing + KNN configuration."""
    numeric_selector = make_column_selector(dtype_include=np.number)  # type: ignore[arg-type]
    categorical_selector = make_column_selector(dtype_include=object)  # type: ignore[arg-type]

    preprocessor = make_column_transformer(
        (Pipeline([("scaler", scaler_cls())]), numeric_selector),
        (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_selector,
        ),
    )

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("knn", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "knn__n_neighbors": [100, 200, 300, 400, 500, 600],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(features_df, y_true)

    return {
        "setting": setting_name,
        "best_score": search.best_score_,
        "best_k": search.best_params_["knn__n_neighbors"],
        "scaler": scaler_cls.__name__,
        "uses_pdays": "Without" not in setting_name,
    }


settings = [
    ("With pdays + StandardScaler", df_train_with_pdays, StandardScaler),
    ("With pdays + Normalizer", df_train_with_pdays, Normalizer),
    ("Without pdays + StandardScaler", df_train_without_pdays, StandardScaler),
    ("Without pdays + Normalizer", df_train_without_pdays, Normalizer),
]

results = []
for setting_name, df_features, scaler_cls in settings:
    print(f"\nRunning grid search for setting: {setting_name}")
    setting_result = evaluate_setting(df_features, scaler_cls, setting_name)
    print(
        f"  -> Best k={setting_result['best_k']} | ROC-AUC={setting_result['best_score']:.4f}"
    )
    results.append(setting_result)

print("\nSummary of settings (mean CV ROC-AUC):")
for res in results:
    print(
        f"  {res['setting']:<35} k={res['best_k']:>3} | score={res['best_score']:.4f}"
    )

best_setting = max(results, key=lambda item: item["best_score"])
print(
    f"\nBest overall setting: {best_setting['setting']} (k={best_setting['best_k']},"
    f" ROC-AUC={best_setting['best_score']:.4f})"
)

without_normalizer = next(
    (res for res in results if res["setting"] == "Without pdays + Normalizer"),
    None,
)
if without_normalizer:
    print(
        f"Best score for 'Without pdays + Normalizer': "
        f"k={without_normalizer['best_k']} | ROC-AUC={without_normalizer['best_score']:.4f}"
    )

# Compare scaler impact within each pdays choice
for uses_pdays, label in [(True, "With pdays"), (False, "Without pdays")]:
    subset = [res for res in results if res["uses_pdays"] == uses_pdays]
    if len(subset) == 2:
        best_scaler = max(subset, key=lambda item: item["best_score"])
        other = min(subset, key=lambda item: item["best_score"])
        diff = best_scaler["best_score"] - other["best_score"]
        print(
            f"{label}: {best_scaler['scaler']} outperforms {other['scaler']} by {diff:.4f} ROC-AUC."
        )

# Compare inclusion vs exclusion of pdays for each scaler choice
for scaler_cls, scaler_label in [
    (StandardScaler, "StandardScaler"),
    (Normalizer, "Normalizer"),
]:
    with_res = next(
        res
        for res in results
        if res["uses_pdays"] and res["scaler"] == scaler_cls.__name__
    )
    without_res = next(
        res
        for res in results
        if (not res["uses_pdays"]) and res["scaler"] == scaler_cls.__name__
    )
    better = (
        with_res if with_res["best_score"] >= without_res["best_score"] else without_res
    )
    comparison = "including" if better["uses_pdays"] else "excluding"
    diff = abs(with_res["best_score"] - without_res["best_score"])
    print(
        f"{scaler_label}: {comparison} 'pdays' is better by {diff:.4f} ROC-AUC"
        f" (with={with_res['best_score']:.4f}, without={without_res['best_score']:.4f})."
    )
