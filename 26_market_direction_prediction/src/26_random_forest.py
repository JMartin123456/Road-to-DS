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
# Model
# ====================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)


# ====================================
# Training
# ====================================

model.fit(
    X_train,
    y_train
)


# ====================================
# Predictions
# ====================================

validation_predictions = model.predict(
    X_validation
)

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]


test_predictions = model.predict(
    X_test
)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


# ====================================
# Evaluation function
# ====================================

def evaluate_model(
    y_true,
    predictions,
    probabilities,
    dataset_name
):

    print(f"\n{'=' * 40}")
    print(dataset_name)
    print(f"{'=' * 40}")

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions
    )

    recall = recall_score(
        y_true,
        predictions
    )

    f1 = f1_score(
        y_true,
        predictions
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )


# ====================================
# Evaluation
# ====================================

evaluate_model(
    y_validation,
    validation_predictions,
    validation_probabilities,
    "Validation"
)

evaluate_model(
    y_test,
    test_predictions,
    test_probabilities,
    "Test"
)


# ====================================
# Feature importance
# ====================================

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature importance:")

print(
    feature_importance
)


# ====================================
# Summary
# ====================================

print("\nRandom Forest model completed.")