from pathlib import Path

import pandas as pd

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


print("=" * 70)
print("LOADING ADVANCED DATASETS")
print("=" * 70)


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

print("\n" + "=" * 70)
print("FEATURE CHECK")
print("=" * 70)

missing_features = [
    feature
    for feature in feature_columns
    if feature not in train.columns
]

if missing_features:

    print("\nMissing features:")
    print(missing_features)

    raise ValueError(
        "Some feature columns are missing."
    )

print(
    f"\nNumber of features: {len(feature_columns)}"
)

print("All advanced features are present.")


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
    model,
    X,
    y,
    dataset_name,
):
    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    print("\n" + "-" * 60)
    print(dataset_name)
    print("-" * 60)

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
# Random Forest tuning
# ====================================

print("\n")
print("=" * 70)
print("RANDOM FOREST TUNING")
print("=" * 70)


random_forest_parameters = [

    {
        "n_estimators": 200,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },

    {
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },

    {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },

    {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },

    {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },

    {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },

    {
        "n_estimators": 500,
        "max_depth": 3,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },

    {
        "n_estimators": 500,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
]


rf_results = []

best_rf_model = None
best_rf_parameters = None
best_rf_validation_auc = -1


for parameters in random_forest_parameters:

    print("\nTesting parameters:")
    print(parameters)

    model = RandomForestClassifier(
        **parameters,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
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

    rf_results.append({
        **parameters,
        "Validation_Accuracy": accuracy,
        "Validation_Precision": precision,
        "Validation_Recall": recall,
        "Validation_F1": f1,
        "Validation_ROC_AUC": roc_auc,
    })

    if roc_auc > best_rf_validation_auc:

        best_rf_validation_auc = roc_auc

        best_rf_parameters = parameters.copy()

        best_rf_model = model


# ====================================
# Random Forest results
# ====================================

rf_results_df = pd.DataFrame(
    rf_results
)

rf_results_df = rf_results_df.sort_values(
    "Validation_ROC_AUC",
    ascending=False,
)


print("\n")
print("=" * 70)
print("RANDOM FOREST TUNING RESULTS")
print("=" * 70)

print(
    rf_results_df.to_string(
        index=False
    )
)


# ====================================
# Best Random Forest
# ====================================

print("\n")
print("=" * 70)
print("BEST RANDOM FOREST")
print("=" * 70)

print(
    "Parameters:",
    best_rf_parameters,
)

print(
    f"Validation ROC-AUC: "
    f"{best_rf_validation_auc:.4f}"
)


# ====================================
# Gradient Boosting tuning
# ====================================

print("\n")
print("=" * 70)
print("GRADIENT BOOSTING TUNING")
print("=" * 70)


gradient_boosting_parameters = [

    {
        "n_estimators": 100,
        "learning_rate": 0.01,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },

    {
        "n_estimators": 200,
        "learning_rate": 0.01,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },

    {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 2,
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
        "n_estimators": 300,
        "learning_rate": 0.03,
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
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },

    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
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
        "max_depth": 3,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },

    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
]


gb_results = []

best_gb_model = None
best_gb_parameters = None
best_gb_validation_auc = -1


for parameters in gradient_boosting_parameters:

    print("\nTesting parameters:")
    print(parameters)

    model = GradientBoostingClassifier(
        **parameters,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
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

    gb_results.append({
        **parameters,
        "Validation_Accuracy": accuracy,
        "Validation_Precision": precision,
        "Validation_Recall": recall,
        "Validation_F1": f1,
        "Validation_ROC_AUC": roc_auc,
    })

    if roc_auc > best_gb_validation_auc:

        best_gb_validation_auc = roc_auc

        best_gb_parameters = parameters.copy()

        best_gb_model = model


# ====================================
# Gradient Boosting results
# ====================================

gb_results_df = pd.DataFrame(
    gb_results
)

gb_results_df = gb_results_df.sort_values(
    "Validation_ROC_AUC",
    ascending=False,
)


print("\n")
print("=" * 70)
print("GRADIENT BOOSTING TUNING RESULTS")
print("=" * 70)

print(
    gb_results_df.to_string(
        index=False
    )
)


# ====================================
# Best Gradient Boosting
# ====================================

print("\n")
print("=" * 70)
print("BEST GRADIENT BOOSTING")
print("=" * 70)

print(
    "Parameters:",
    best_gb_parameters,
)

print(
    f"Validation ROC-AUC: "
    f"{best_gb_validation_auc:.4f}"
)


# ====================================
# Evaluate best models
# ====================================

print("\n")
print("=" * 70)
print("BEST RANDOM FOREST - VALIDATION")
print("=" * 70)

rf_validation_metrics = evaluate_model(
    best_rf_model,
    X_validation,
    y_validation,
    "Random Forest Validation",
)


print("\n")
print("=" * 70)
print("BEST RANDOM FOREST - TEST")
print("=" * 70)

rf_test_metrics = evaluate_model(
    best_rf_model,
    X_test,
    y_test,
    "Random Forest Test",
)


print("\n")
print("=" * 70)
print("BEST GRADIENT BOOSTING - VALIDATION")
print("=" * 70)

gb_validation_metrics = evaluate_model(
    best_gb_model,
    X_validation,
    y_validation,
    "Gradient Boosting Validation",
)


print("\n")
print("=" * 70)
print("BEST GRADIENT BOOSTING - TEST")
print("=" * 70)

gb_test_metrics = evaluate_model(
    best_gb_model,
    X_test,
    y_test,
    "Gradient Boosting Test",
)


# ====================================
# Feature importance
# ====================================

print("\n")
print("=" * 70)
print("BEST RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

rf_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": best_rf_model.feature_importances_,
})

rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False,
).reset_index(
    drop=True
)

print(
    rf_importance.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("BEST GRADIENT BOOSTING FEATURE IMPORTANCE")
print("=" * 70)

gb_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": best_gb_model.feature_importances_,
})

gb_importance = gb_importance.sort_values(
    "Importance",
    ascending=False,
).reset_index(
    drop=True
)

print(
    gb_importance.to_string(
        index=False
    )
)


# ====================================
# Model comparison
# ====================================

comparison = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Validation_Accuracy":
            rf_validation_metrics["Accuracy"],
        "Validation_Precision":
            rf_validation_metrics["Precision"],
        "Validation_Recall":
            rf_validation_metrics["Recall"],
        "Validation_F1":
            rf_validation_metrics["F1"],
        "Validation_ROC_AUC":
            rf_validation_metrics["ROC_AUC"],
        "Test_Accuracy":
            rf_test_metrics["Accuracy"],
        "Test_Precision":
            rf_test_metrics["Precision"],
        "Test_Recall":
            rf_test_metrics["Recall"],
        "Test_F1":
            rf_test_metrics["F1"],
        "Test_ROC_AUC":
            rf_test_metrics["ROC_AUC"],
    },

    {
        "Model": "Gradient Boosting",
        "Validation_Accuracy":
            gb_validation_metrics["Accuracy"],
        "Validation_Precision":
            gb_validation_metrics["Precision"],
        "Validation_Recall":
            gb_validation_metrics["Recall"],
        "Validation_F1":
            gb_validation_metrics["F1"],
        "Validation_ROC_AUC":
            gb_validation_metrics["ROC_AUC"],
        "Test_Accuracy":
            gb_test_metrics["Accuracy"],
        "Test_Precision":
            gb_test_metrics["Precision"],
        "Test_Recall":
            gb_test_metrics["Recall"],
        "Test_F1":
            gb_test_metrics["F1"],
        "Test_ROC_AUC":
            gb_test_metrics["ROC_AUC"],
    },
])


# ====================================
# Save results
# ====================================

rf_results_file = (
    PROCESSED_DATA_DIR
    / "advanced_random_forest_tuning.csv"
)

gb_results_file = (
    PROCESSED_DATA_DIR
    / "advanced_gradient_boosting_tuning.csv"
)

comparison_file = (
    PROCESSED_DATA_DIR
    / "advanced_tuning_model_comparison.csv"
)


rf_results_df.to_csv(
    rf_results_file,
    index=False,
)

gb_results_df.to_csv(
    gb_results_file,
    index=False,
)

comparison.to_csv(
    comparison_file,
    index=False,
)


# ====================================
# Final summary
# ====================================

print("\n")
print("=" * 70)
print("ADVANCED MODEL TUNING SUMMARY")
print("=" * 70)

print("\nBest Random Forest parameters:")

print(
    best_rf_parameters
)

print(
    f"RF Validation ROC-AUC: "
    f"{best_rf_validation_auc:.4f}"
)

print(
    f"RF Test ROC-AUC: "
    f"{rf_test_metrics['ROC_AUC']:.4f}"
)


print("\nBest Gradient Boosting parameters:")

print(
    best_gb_parameters
)

print(
    f"GB Validation ROC-AUC: "
    f"{best_gb_validation_auc:.4f}"
)

print(
    f"GB Test ROC-AUC: "
    f"{gb_test_metrics['ROC_AUC']:.4f}"
)


print("\nResults saved:")

print(
    rf_results_file
)

print(
    gb_results_file
)

print(
    comparison_file
)


print("\n")
print("=" * 70)
print("ADVANCED MODEL TUNING COMPLETED")
print("=" * 70)