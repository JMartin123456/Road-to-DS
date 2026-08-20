from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

REPORTS_DIR = (
    BASE_DIR
    / "reports"
)

FIGURES_DIR = (
    REPORTS_DIR
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = (
    PROCESSED_DATA_DIR
    / "train_advanced.csv"
)

VALIDATION_FILE = (
    PROCESSED_DATA_DIR
    / "validation_advanced.csv"
)

TEST_FILE = (
    PROCESSED_DATA_DIR
    / "test_advanced.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING ADVANCED DATASETS")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print("\nTrain shape:")
print(train.shape)

print("\nValidation shape:")
print(validation.shape)

print("\nTest shape:")
print(test.shape)


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in train.columns
    if column not in [
        "Date",
        "Target",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
]


print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(f"\nNumber of features: {len(feature_columns)}")

print("\nFeatures:")

for index, feature in enumerate(
    feature_columns,
    start=1
):
    print(
        f"{index:02d}. {feature}"
    )


# ============================================================
# X / y
# ============================================================

X_train = train[feature_columns]
y_train = train["Target"]

X_validation = validation[feature_columns]
y_validation = validation["Target"]

X_test = test[feature_columns]
y_test = test["Target"]


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)


def print_target_distribution(
    y,
    name
):

    counts = y.value_counts()

    percentages = (
        y.value_counts(
            normalize=True
        )
        * 100
    )

    print(f"\n{name}")

    print(
        pd.DataFrame({
            "Count": counts,
            "Percentage": percentages
        })
    )


print_target_distribution(
    y_train,
    "Train"
)

print_target_distribution(
    y_validation,
    "Validation"
)

print_target_distribution(
    y_test,
    "Test"
)


# ============================================================
# 2. NAIVE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("NAIVE BASELINE")
print("=" * 70)

majority_class = (
    y_train
    .value_counts()
    .idxmax()
)

print(
    f"\nMajority class: {majority_class}"
)

validation_naive = np.full(
    len(y_validation),
    majority_class
)

test_naive = np.full(
    len(y_test),
    majority_class
)

print(
    f"\nNaive Validation Accuracy: "
    f"{accuracy_score(y_validation, validation_naive):.4f}"
)

print(
    f"Naive Test Accuracy: "
    f"{accuracy_score(y_test, test_naive):.4f}"
)


# ============================================================
# 3. FEATURE CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CORRELATION WITH TARGET")
print("=" * 70)

correlation_data = train[
    feature_columns + ["Target"]
].corr()

target_correlation = (
    correlation_data["Target"]
    .drop("Target")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

correlation_results = pd.DataFrame({
    "Feature": target_correlation.index,
    "Correlation": target_correlation.values,
    "Absolute_Correlation":
        target_correlation.abs().values
})


print("\nTop features by absolute correlation:")

print(
    correlation_results.to_string(
        index=False
    )
)


correlation_output = (
    PROCESSED_DATA_DIR
    / "diagnostics_feature_target_correlation.csv"
)

correlation_results.to_csv(
    correlation_output,
    index=False
)


# ============================================================
# 4. FEATURE STATISTICS BY TARGET
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS BY TARGET")
print("=" * 70)

grouped_statistics = []

for feature in feature_columns:

    mean_down = train.loc[
        train["Target"] == 0,
        feature
    ].mean()

    mean_up = train.loc[
        train["Target"] == 1,
        feature
    ].mean()

    std_down = train.loc[
        train["Target"] == 0,
        feature
    ].std()

    std_up = train.loc[
        train["Target"] == 1,
        feature
    ].std()

    difference = (
        mean_up
        - mean_down
    )

    grouped_statistics.append({
        "Feature": feature,
        "Mean_DOWN": mean_down,
        "Mean_UP": mean_up,
        "Difference": difference,
        "Std_DOWN": std_down,
        "Std_UP": std_up,
    })


grouped_statistics = pd.DataFrame(
    grouped_statistics
)

grouped_statistics[
    "Absolute_Difference"
] = (
    grouped_statistics[
        "Difference"
    ]
    .abs()
)

grouped_statistics = (
    grouped_statistics
    .sort_values(
        "Absolute_Difference",
        ascending=False
    )
)


print(
    grouped_statistics.to_string(
        index=False
    )
)


statistics_output = (
    PROCESSED_DATA_DIR
    / "diagnostics_feature_statistics.csv"
)

grouped_statistics.to_csv(
    statistics_output,
    index=False
)


# ============================================================
# 5. STANDARDIZED MEAN DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZED FEATURE DIFFERENCE")
print("=" * 70)

effect_results = []

for feature in feature_columns:

    down_values = train.loc[
        train["Target"] == 0,
        feature
    ].dropna()

    up_values = train.loc[
        train["Target"] == 1,
        feature
    ].dropna()

    mean_down = down_values.mean()
    mean_up = up_values.mean()

    std_down = down_values.std()
    std_up = up_values.std()

    pooled_std = np.sqrt(
        (
            std_down ** 2
            +
            std_up ** 2
        )
        / 2
    )

    if pooled_std == 0:

        effect_size = 0

    else:

        effect_size = (
            mean_up
            - mean_down
        ) / pooled_std

    effect_results.append({
        "Feature": feature,
        "Effect_Size": effect_size,
        "Absolute_Effect_Size":
            abs(effect_size)
    })


effect_results = pd.DataFrame(
    effect_results
)

effect_results = (
    effect_results
    .sort_values(
        "Absolute_Effect_Size",
        ascending=False
    )
)


print(
    effect_results.to_string(
        index=False
    )
)


effect_output = (
    PROCESSED_DATA_DIR
    / "diagnostics_effect_sizes.csv"
)

effect_results.to_csv(
    effect_output,
    index=False
)


# ============================================================
# 6. LOGISTIC REGRESSION FEATURE SIGNAL
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION FEATURE SIGNAL")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = (
    scaler.transform(
        X_validation
    )
)

logistic_model = LogisticRegression(
    max_iter=2000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_validation_probability = (
    logistic_model
    .predict_proba(
        X_validation_scaled
    )[:, 1]
)

logistic_validation_prediction = (
    logistic_model
    .predict(
        X_validation_scaled
    )
)

logistic_auc = roc_auc_score(
    y_validation,
    logistic_validation_probability
)

logistic_accuracy = accuracy_score(
    y_validation,
    logistic_validation_prediction
)

print(
    f"\nValidation Accuracy: "
    f"{logistic_accuracy:.4f}"
)

print(
    f"Validation ROC-AUC: "
    f"{logistic_auc:.4f}"
)


logistic_coefficients = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient":
        logistic_model.coef_[0]
})

logistic_coefficients[
    "Absolute_Coefficient"
] = (
    logistic_coefficients[
        "Coefficient"
    ]
    .abs()
)

logistic_coefficients = (
    logistic_coefficients
    .sort_values(
        "Absolute_Coefficient",
        ascending=False
    )
)


print("\nLogistic Regression coefficients:")

print(
    logistic_coefficients.to_string(
        index=False
    )
)


logistic_output = (
    PROCESSED_DATA_DIR
    / "diagnostics_logistic_coefficients.csv"
)

logistic_coefficients.to_csv(
    logistic_output,
    index=False
)


# ============================================================
# 7. RANDOM FOREST PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST PERMUTATION IMPORTANCE")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_validation_probability = (
    rf_model
    .predict_proba(
        X_validation
    )[:, 1]
)

rf_validation_prediction = (
    rf_model
    .predict(
        X_validation
    )
)

rf_auc = roc_auc_score(
    y_validation,
    rf_validation_probability
)

rf_accuracy = accuracy_score(
    y_validation,
    rf_validation_prediction
)

print(
    f"\nRandom Forest Validation Accuracy: "
    f"{rf_accuracy:.4f}"
)

print(
    f"Random Forest Validation ROC-AUC: "
    f"{rf_auc:.4f}"
)


permutation = permutation_importance(
    rf_model,
    X_validation,
    y_validation,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_results = pd.DataFrame({
    "Feature": feature_columns,
    "Importance_Mean":
        permutation.importances_mean,
    "Importance_Std":
        permutation.importances_std
})

permutation_results[
    "Absolute_Importance"
] = (
    permutation_results[
        "Importance_Mean"
    ]
    .abs()
)

permutation_results = (
    permutation_results
    .sort_values(
        "Importance_Mean",
        ascending=False
    )
)


print(
    permutation_results.to_string(
        index=False
    )
)


permutation_output = (
    PROCESSED_DATA_DIR
    / "diagnostics_permutation_importance.csv"
)

permutation_results.to_csv(
    permutation_output,
    index=False
)


# ============================================================
# 8. GRADIENT BOOSTING DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING DIAGNOSTICS")
print("=" * 70)

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

gb_model.fit(
    X_train,
    y_train
)

gb_validation_probability = (
    gb_model
    .predict_proba(
        X_validation
    )[:, 1]
)

gb_validation_prediction = (
    gb_model
    .predict(
        X_validation
    )
)

gb_auc = roc_auc_score(
    y_validation,
    gb_validation_probability
)

gb_accuracy = accuracy_score(
    y_validation,
    gb_validation_prediction
)

print(
    f"\nGradient Boosting Validation Accuracy: "
    f"{gb_accuracy:.4f}"
)

print(
    f"Gradient Boosting Validation ROC-AUC: "
    f"{gb_auc:.4f}"
)


gb_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance":
        gb_model.feature_importances_
})

gb_importance = (
    gb_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print("\nGradient Boosting feature importance:")

print(
    gb_importance.to_string(
        index=False
    )
)


gb_output = (
    PROCESSED_DATA_DIR
    / "diagnostics_gradient_boosting_importance.csv"
)

gb_importance.to_csv(
    gb_output,
    index=False
)


# ============================================================
# 9. TRAIN VS VALIDATION VS TEST
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION / TEST FEATURE DISTRIBUTION")
print("=" * 70)

distribution_results = []

for feature in feature_columns:

    train_mean = X_train[feature].mean()
    validation_mean = X_validation[feature].mean()
    test_mean = X_test[feature].mean()

    train_std = X_train[feature].std()
    validation_std = X_validation[feature].std()
    test_std = X_test[feature].std()

    distribution_results.append({
        "Feature": feature,
        "Train_Mean": train_mean,
        "Validation_Mean": validation_mean,
        "Test_Mean": test_mean,
        "Train_Std": train_std,
        "Validation_Std": validation_std,
        "Test_Std": test_std,
        "Validation_Mean_Diff":
            abs(
                train_mean
                - validation_mean
            ),
        "Test_Mean_Diff":
            abs(
                train_mean
                - test_mean
            )
    })


distribution_results = pd.DataFrame(
    distribution_results
)

distribution_results = (
    distribution_results
    .sort_values(
        "Test_Mean_Diff",
        ascending=False
    )
)


print(
    distribution_results.to_string(
        index=False
    )
)


distribution_output = (
    PROCESSED_DATA_DIR
    / "diagnostics_distribution_shift.csv"
)

distribution_results.to_csv(
    distribution_output,
    index=False
)


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL DIAGNOSTICS SUMMARY")
print("=" * 70)

summary = pd.DataFrame({
    "Model": [
        "Naive Baseline",
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Validation_Accuracy": [
        accuracy_score(
            y_validation,
            validation_naive
        ),
        logistic_accuracy,
        rf_accuracy,
        gb_accuracy
    ],
    "Validation_ROC_AUC": [
        0.5,
        logistic_auc,
        rf_auc,
        gb_auc
    ]
})


print(
    summary.to_string(
        index=False
    )
)


summary_output = (
    PROCESSED_DATA_DIR
    / "diagnostics_model_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)


# ============================================================
# 11. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DIAGNOSTIC INTERPRETATION")
print("=" * 70)

best_auc = max(
    logistic_auc,
    rf_auc,
    gb_auc
)

print(
    f"\nBest Validation ROC-AUC: "
    f"{best_auc:.4f}"
)

if best_auc < 0.52:

    print(
        "\nWARNING:"
    )

    print(
        "The current models show very weak "
        "predictive signal."
    )

    print(
        "ROC-AUC is close to random guessing."
    )

    print(
        "Further model complexity alone is "
        "unlikely to solve the problem."
    )

elif best_auc < 0.55:

    print(
        "\nWEAK SIGNAL:"
    )

    print(
        "There may be a small predictive signal, "
        "but it is weak."
    )

else:

    print(
        "\nPOTENTIAL SIGNAL:"
    )

    print(
        "The model shows a potentially useful "
        "predictive signal."
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC FILES SAVED")
print("=" * 70)

print(
    f"\n{correlation_output}"
)

print(
    f"{statistics_output}"
)

print(
    f"{effect_output}"
)

print(
    f"{logistic_output}"
)

print(
    f"{permutation_output}"
)

print(
    f"{gb_output}"
)

print(
    f"{distribution_output}"
)

print(
    f"{summary_output}"
)

print("\n")
print("=" * 70)
print("MODEL DIAGNOSTICS COMPLETED")
print("=" * 70)