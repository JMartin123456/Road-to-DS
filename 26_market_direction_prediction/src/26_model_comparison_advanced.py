from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

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

print("\n========================================")
print("Loading advanced datasets")
print("========================================")

train_file = (
    PROCESSED_DATA_DIR
    / "train_advanced.csv"
)

validation_file = (
    PROCESSED_DATA_DIR
    / "validation_advanced.csv"
)

test_file = (
    PROCESSED_DATA_DIR
    / "test_advanced.csv"
)


train = pd.read_csv(train_file)
validation = pd.read_csv(validation_file)
test = pd.read_csv(test_file)


print("\nTrain shape:")
print(train.shape)

print("\nValidation shape:")
print(validation.shape)

print("\nTest shape:")
print(test.shape)


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

    "Momentum_3d",
    "Momentum_10d",
    "Momentum_30d",
    "Momentum_Acceleration",
    "Return_5d_change",

    "Price_vs_SMA_5",
    "Price_vs_SMA_50",

    "SMA_5_vs_SMA_20",
    "SMA_20_vs_SMA_50",

    "SMA_5_slope",
    "SMA_20_slope",
    "SMA_50_slope",

    "Volatility_5_vs_20",
    "Volatility_20_vs_50",
    "Volatility_5_change",
    "Volatility_20_change",

    "Range_5d",
    "Range_20d",
    "Range_50d",

    "Price_Position_20",
    "Price_Position_50",

    "Distance_From_High_20",
    "Distance_From_Low_20",

    "Body_to_Range",
    "Upper_Wick_to_Range",
    "Lower_Wick_to_Range",

    "Candle_Direction",
    "Previous_Candle_Direction",
    "Previous_Return",

    "Return_Mean_5",
    "Return_Mean_20",

    "Return_Std_5",
    "Return_Std_20",

    "Positive_Return_Ratio_10",
    "Positive_Return_Ratio_20",

    "Momentum_5_Vol_Adjusted",
    "Momentum_20_Vol_Adjusted",
]


# ====================================
# Feature check
# ====================================

print("\n========================================")
print("Feature check")
print("========================================")

print(
    f"\nNumber of features: {len(feature_columns)}"
)


missing_features = [
    feature
    for feature in feature_columns
    if feature not in train.columns
]


if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print(feature)

    raise ValueError(
        "Some feature columns are missing."
    )


print("\nAll advanced features are present.")


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
# Evaluation function
# ====================================

def evaluate_model(
    y_true,
    predictions,
    probabilities,
    dataset_name,
):

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    print(
        f"\n{dataset_name}"
    )

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
            y_true,
            predictions,
        )
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
    }


# ====================================
# Results storage
# ====================================

validation_results = []
test_results = []


# ====================================
# 1. Logistic Regression
# ====================================

print("\n")
print("=" * 70)
print("Training: Logistic Regression")
print("=" * 70)


logistic_model = LogisticRegression(
    max_iter=2000,
)


logistic_model.fit(
    X_train,
    y_train,
)


# Validation

validation_predictions = (
    logistic_model.predict(
        X_validation
    )
)

validation_probabilities = (
    logistic_model.predict_proba(
        X_validation
    )[:, 1]
)


validation_metrics = evaluate_model(
    y_validation,
    validation_predictions,
    validation_probabilities,
    "Logistic Regression - Validation",
)


validation_results.append(
    {
        "Model": "Logistic Regression",
        **validation_metrics,
    }
)


# Test

test_predictions = (
    logistic_model.predict(
        X_test
    )
)

test_probabilities = (
    logistic_model.predict_proba(
        X_test
    )[:, 1]
)


test_metrics = evaluate_model(
    y_test,
    test_predictions,
    test_probabilities,
    "Logistic Regression - Test",
)


test_results.append(
    {
        "Model": "Logistic Regression",
        **test_metrics,
    }
)


# ====================================
# 2. Random Forest
# ====================================

print("\n")
print("=" * 70)
print("Training: Random Forest")
print("=" * 70)


random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)


random_forest_model.fit(
    X_train,
    y_train,
)


# Validation

validation_predictions = (
    random_forest_model.predict(
        X_validation
    )
)

validation_probabilities = (
    random_forest_model.predict_proba(
        X_validation
    )[:, 1]
)


validation_metrics = evaluate_model(
    y_validation,
    validation_predictions,
    validation_probabilities,
    "Random Forest - Validation",
)


validation_results.append(
    {
        "Model": "Random Forest",
        **validation_metrics,
    }
)


# Test

test_predictions = (
    random_forest_model.predict(
        X_test
    )
)

test_probabilities = (
    random_forest_model.predict_proba(
        X_test
    )[:, 1]
)


test_metrics = evaluate_model(
    y_test,
    test_predictions,
    test_probabilities,
    "Random Forest - Test",
)


test_results.append(
    {
        "Model": "Random Forest",
        **test_metrics,
    }
)


# ====================================
# 3. Gradient Boosting
# ====================================

print("\n")
print("=" * 70)
print("Training: Gradient Boosting")
print("=" * 70)


gradient_boosting_model = (
    GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
)


gradient_boosting_model.fit(
    X_train,
    y_train,
)


# Validation

validation_predictions = (
    gradient_boosting_model.predict(
        X_validation
    )
)

validation_probabilities = (
    gradient_boosting_model.predict_proba(
        X_validation
    )[:, 1]
)


validation_metrics = evaluate_model(
    y_validation,
    validation_predictions,
    validation_probabilities,
    "Gradient Boosting - Validation",
)


validation_results.append(
    {
        "Model": "Gradient Boosting",
        **validation_metrics,
    }
)


# Test

test_predictions = (
    gradient_boosting_model.predict(
        X_test
    )
)

test_probabilities = (
    gradient_boosting_model.predict_proba(
        X_test
    )[:, 1]
)


test_metrics = evaluate_model(
    y_test,
    test_predictions,
    test_probabilities,
    "Gradient Boosting - Test",
)


test_results.append(
    {
        "Model": "Gradient Boosting",
        **test_metrics,
    }
)


# ====================================
# Results DataFrames
# ====================================

validation_results_df = pd.DataFrame(
    validation_results
)

test_results_df = pd.DataFrame(
    test_results
)


# ====================================
# Sort results
# ====================================

validation_results_df = (
    validation_results_df
    .sort_values(
        "ROC_AUC",
        ascending=False,
    )
    .reset_index(drop=True)
)


test_results_df = (
    test_results_df
    .sort_values(
        "ROC_AUC",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ====================================
# Validation comparison
# ====================================

print("\n")
print("=" * 70)
print("ADVANCED VALIDATION MODEL COMPARISON")
print("=" * 70)

print(
    validation_results_df.to_string(
        index=False
    )
)


# ====================================
# Test comparison
# ====================================

print("\n")
print("=" * 70)
print("ADVANCED TEST MODEL COMPARISON")
print("=" * 70)

print(
    test_results_df.to_string(
        index=False
    )
)


# ====================================
# Best validation model
# ====================================

best_validation_model = (
    validation_results_df.iloc[0]
)


print("\n")
print("=" * 70)
print("BEST ADVANCED VALIDATION MODEL")
print("=" * 70)

print(
    best_validation_model.to_string()
)


# ====================================
# Best test model
# ====================================

best_test_model = (
    test_results_df.iloc[0]
)


print("\n")
print("=" * 70)
print("BEST ADVANCED TEST MODEL")
print("=" * 70)

print(
    best_test_model.to_string()
)


# ====================================
# Compare against original models
# ====================================

print("\n")
print("=" * 70)
print("ADVANCED MODEL COMPARISON COMPLETED")
print("=" * 70)


# ====================================
# Save results
# ====================================

validation_output = (
    PROCESSED_DATA_DIR
    / "model_comparison_advanced_validation.csv"
)

test_output = (
    PROCESSED_DATA_DIR
    / "model_comparison_advanced_test.csv"
)


validation_results_df.to_csv(
    validation_output,
    index=False,
)

test_results_df.to_csv(
    test_output,
    index=False,
)


print("\nResults saved:")

print(
    validation_output
)

print(
    test_output
)


print("\n")
print("=" * 70)
print("ADVANCED MODEL COMPARISON FINISHED")
print("=" * 70)