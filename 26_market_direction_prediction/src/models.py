"""Machine learning models for the EUR/USD prediction project."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets import check_features
from evaluation import metrics

RANDOM_STATE = 42

def model_factories() -> Mapping[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        ),
    }

def compare_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_list: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    feature_list = list(feature_list)
    check_features((train, validation, test), feature_list)
    X_train, y_train = train[feature_list], train["Target"]
    results = {"Validation": [], "Test": []}
    probabilities = {"Validation": {}, "Test": {}}
    fitted = {}
    for name, model in model_factories().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        for dataset_name, frame in (("Validation", validation), ("Test", test)):
            probabilities[dataset_name][name] = model.predict_proba(frame[feature_list])[:, 1]
            predictions = (probabilities[dataset_name][name] >= 0.5).astype(int)
            row = {"Model": name, **metrics(frame["Target"], predictions, probabilities[dataset_name][name])}
            results[dataset_name].append(row)
    validation_result = pd.DataFrame(results["Validation"]).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    test_result = pd.DataFrame(results["Test"]).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return validation_result, test_result, probabilities, fitted

def tune_model(
    estimator_class: type, parameter_combinations: Iterable[dict],
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    feature_list: Iterable[str], model_name: str,
) -> tuple[pd.DataFrame, object, dict, dict]:
    feature_list = list(feature_list)
    check_features((train, validation, test), feature_list)
    rows, best_model, best_params, best_auc = [], None, None, -np.inf
    for parameters in parameter_combinations:
        model = estimator_class(**parameters, random_state=RANDOM_STATE, **(
            {"n_jobs": -1} if estimator_class is RandomForestClassifier else {}
        ))
        model.fit(train[feature_list], train["Target"])
        probabilities = model.predict_proba(validation[feature_list])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        row = dict(parameters)
        row.update({f"Validation_{key}": value for key, value in metrics(
            validation["Target"], predictions, probabilities
        ).items()})
        rows.append(row)
        if row["Validation_ROC_AUC"] > best_auc:
            best_auc, best_model, best_params = row["Validation_ROC_AUC"], model, dict(parameters)
    result = pd.DataFrame(rows).sort_values("Validation_ROC_AUC", ascending=False).reset_index(drop=True)
    final = {}
    for name, frame in (("Validation", validation), ("Test", test)):
        probabilities = best_model.predict_proba(frame[feature_list])[:, 1]
        final[name] = metrics(frame["Target"], (probabilities >= 0.5).astype(int), probabilities)
    return result, best_model, best_params, final