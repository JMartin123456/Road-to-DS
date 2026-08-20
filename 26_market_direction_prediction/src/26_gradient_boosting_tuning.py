from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

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

train_file = PROCESSED_DATA_DIR / "train.csv"
validation_file = PROCESSED_DATA_DIR / "validation.csv"
test_file = PROCESSED_DATA_DIR / "test.csv"

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
# Parameter combinations
# ====================================

parameter_combinations = [
    {
        "n_estimators": 100,
        "learning_rate": 0.01,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
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

    model = GradientBoostingClassifier(
        n_estimators=parameters["n_estimators"],
        learning_rate=parameters["learning_rate"],
        max_depth=parameters["max_depth"],
        min_samples_split=parameters["min_samples_split"],
        min_samples_leaf=parameters["min_samples_leaf"],
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    validation_predictions = model.predict(
        X_validation
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    precision = precision_score(
        y_validation,
        validation_predictions
    )

    recall = recall_score(
        y_validation,
        validation_predictions
    )

    f1 = f1_score(
        y_validation,
        validation_predictions
    )

    roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities
    )

    print(
        f"Validation Accuracy: {accuracy:.4f}"
    )

    print(
        f"Validation F1:       {f1:.4f}"
    )

    print(
        f"Validation ROC-AUC:  {roc_auc:.4f}"
    )

    results.append({
        **parameters,
        "Validation_Accuracy": accuracy,
        "Validation_Precision": precision,
        "Validation_Recall": recall,
        "Validation_F1": f1,
        "Validation_ROC_AUC": roc_auc,
    })

    if roc_auc > best_validation_auc:

        best_validation_auc = roc_auc
        best_model = model
        best_parameters = parameters


# ====================================
# Tuning results
# ====================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Validation_ROC_AUC",
    ascending=False
)


print("\n")
print("=" * 40)
print("Tuning results")
print("=" * 40)

print(
    results_df
)


# ====================================
# Best model
# ====================================

print("\n")
print("=" * 40)
print("Best model")
print("=" * 40)

print(
    "Parameters:",
    best_parameters
)

print(
    f"Validation ROC-AUC: "
    f"{best_validation_auc:.4f}"
)


# ====================================
# Final evaluation function
# ====================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name
):

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions
    )

    recall = recall_score(
        y,
        predictions
    )

    f1 = f1_score(
        y,
        predictions
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    print("\n")
    print("=" * 40)
    print(dataset_name)
    print("=" * 40)

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    print("\nPrediction distribution:")

    print(
        pd.Series(
            predictions
        ).value_counts()
    )


# ====================================
# Best model evaluation
# ====================================

evaluate_model(
    best_model,
    X_validation,
    y_validation,
    "Best model - Validation"
)

evaluate_model(
    best_model,
    X_test,
    y_test,
    "Best model - Test"
)


# ====================================
# Feature importance
# ====================================

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": best_model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)


print("\n")
print("=" * 40)
print("Feature importance")
print("=" * 40)

print(
    feature_importance
)


# ====================================
# Summary
# ====================================

print("\nGradient Boosting tuning completed.")