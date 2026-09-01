"""Evaluation metrics and threshold analysis for the EUR/USD project."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def metrics(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC_AUC": np.nan,
    }
    if y_true.nunique() > 1:
        result["ROC_AUC"] = roc_auc_score(y_true, probabilities)
    return result

def evaluate_thresholds(
    y_true: pd.Series, probabilities: np.ndarray, model_name: str, dataset_name: str,
    thresholds: Iterable[float] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        result = metrics(y_true, predictions, probabilities)
        rows.append({
            "Model": model_name, "Dataset": dataset_name, "Threshold": threshold,
            **{key: result[key] for key in ("Accuracy", "Precision", "Recall", "F1")},
            "Predicted_UP": int(predictions.sum()),
            "Predicted_DOWN": int(len(predictions) - predictions.sum()),
        })
    return pd.DataFrame(rows)