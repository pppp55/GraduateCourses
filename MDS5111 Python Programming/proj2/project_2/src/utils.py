# src/utils.py
# Utility functions of the projects.
# YOU SHOULD NOT MODIFY THIS FILE.

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def coefficient_of_variation(series: pd.Series) -> float:
    mean_val = series.mean()
    std_val = series.std()
    if pd.isna(mean_val) or mean_val == 0:
        return np.nan
    return std_val / mean_val


def mean_absolute_percentage_error_safe(y_true, y_pred, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs(y_true - y_pred) / denom)