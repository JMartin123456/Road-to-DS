from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ====================================
# Config
# ====================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# ====================================
# Load datasets
# ====================================

train_file = (
    PROCESSED_DATA_DIR
    / "train.csv"
)

validation_file = (
    PROCESSED_DATA_DIR
    / "validation.csv"
)

test_file = (
    PROCESSED_DATA_DIR
    / "test.csv"
)

train = pd.read_csv(train_file)
validation = pd.read_csv(validation_file)
test = pd.read_csv(test_file)


# ====================================
# Feature columns
# ====================================

feature_columns = [
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "Volatility_5",
    "Volatility_20",
    "Volatility_50",
    "Volatility_Ratio_5_20",
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "Price_vs_SMA_20",
    "High_Low_Range_Pct",
    "Open_Close_Range_Pct",
    "Body_Size_Pct",
    "Upper_Wick_Pct",
    "Lower_Wick_Pct",
]


# ====================================
# X and y
# ====================================

X_train = train[feature_columns]
y_train = train["Target"]

X_validation = validation[feature_columns]
y_validation = validation["Target"]

X_test = test[feature_columns]
y_test = test["Target"]


# ====================================
# Hyperparameter combinations
# ====================================

parameter_combinations = [
    {
        "n_estimators": 200,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
    {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
    {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
    },
]


# ====================================
# Tuning
# ====================================

results = []

best_model = None
best_parameters = None
best_validation_auc = -1


for parameters in parameter_combinations:

    print("\nTesting parameters:")
    print(parameters)

    model = RandomForestClassifier(
        **parameters,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    validation_predictions = model.predict(
        X_validation
    )

    validation_auc = roc_auc_score(
        y_validation,
        validation_probabilities
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    validation_recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    results.append({
        **parameters,
        "Validation_Accuracy": validation_accuracy,
        "Validation_Precision": validation_precision,
        "Validation_Recall": validation_recall,
        "Validation_F1": validation_f1,
        "Validation_ROC_AUC": validation_auc,
    })

    print(
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )

    print(
        f"Validation F1:       {validation_f1:.4f}"
    )

    print(
        f"Validation ROC-AUC:  {validation_auc:.4f}"
    )

    if validation_auc > best_validation_auc:

        best_validation_auc = validation_auc

        best_model = model

        best_parameters = parameters


# ====================================
# Tuning results
# ====================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Validation_ROC_AUC",
    ascending=False
)

print("\n========================================")
print("Tuning results")
print("========================================")

print(results_df)


# ====================================
# Best model
# ====================================

print("\n========================================")
print("Best model")
print("========================================")

print(
    "Parameters:",
    best_parameters
)

print(
    f"Validation ROC-AUC: "
    f"{best_validation_auc:.4f}"
)


# ====================================
# Final validation evaluation
# ====================================

validation_predictions = best_model.predict(
    X_validation
)

validation_probabilities = best_model.predict_proba(
    X_validation
)[:, 1]

print("\n========================================")
print("Best model - Validation")
print("========================================")

print(
    f"Accuracy:  "
    f"{accuracy_score(y_validation, validation_predictions):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_validation, validation_predictions, zero_division=0):.4f}"
)

print(
    f"Recall:    "
    f"{recall_score(y_validation, validation_predictions, zero_division=0):.4f}"
)

print(
    f"F1 Score:  "
    f"{f1_score(y_validation, validation_predictions, zero_division=0):.4f}"
)

print(
    f"ROC-AUC:   "
    f"{roc_auc_score(y_validation, validation_probabilities):.4f}"
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        y_validation,
        validation_predictions
    )
)


# ====================================
# Test evaluation
# ====================================

test_predictions = best_model.predict(
    X_test
)

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

print("\n========================================")
print("Best model - Test")
print("========================================")

print(
    f"Accuracy:  "
    f"{accuracy_score(y_test, test_predictions):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_test, test_predictions, zero_division=0):.4f}"
)

print(
    f"Recall:    "
    f"{recall_score(y_test, test_predictions, zero_division=0):.4f}"
)

print(
    f"F1 Score:  "
    f"{f1_score(y_test, test_predictions, zero_division=0):.4f}"
)

print(
    f"ROC-AUC:   "
    f"{roc_auc_score(y_test, test_probabilities):.4f}"
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# ====================================
# Feature importance
# ====================================

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": best_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\n========================================")
print("Feature importance")
print("========================================")

print(
    feature_importance
)


# ====================================
# Summary
# ====================================

print("\nRandom Forest tuning completed.")