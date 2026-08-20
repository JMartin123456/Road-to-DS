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

train = pd.read_csv(
    PROCESSED_DATA_DIR / "train.csv"
)

validation = pd.read_csv(
    PROCESSED_DATA_DIR / "validation.csv"
)

test = pd.read_csv(
    PROCESSED_DATA_DIR / "test.csv"
)


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
# Models
# ====================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ),
}


# ====================================
# Thresholds
# ====================================

thresholds = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]


# ====================================
# Evaluation function
# ====================================

def evaluate_thresholds(
    y_true,
    probabilities,
    dataset_name,
    model_name
):

    results = []

    print("\n")
    print("=" * 70)
    print(f"{model_name} - {dataset_name}")
    print("=" * 70)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        predicted_up = predictions.sum()

        predicted_down = (
            len(predictions)
            - predicted_up
        )

        results.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Predicted_UP": predicted_up,
            "Predicted_DOWN": predicted_down,
        })

        print(
            f"Threshold: {threshold:.2f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"UP: {predicted_up} | "
            f"DOWN: {predicted_down}"
        )

    return results


# ====================================
# Train models and calculate
# probabilities
# ====================================

all_results = []

validation_probabilities = {}
test_probabilities = {}


for model_name, model in models.items():

    print("\n")
    print("=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    validation_probabilities[
        model_name
    ] = validation_probability

    test_probabilities[
        model_name
    ] = test_probability

    validation_auc = roc_auc_score(
        y_validation,
        validation_probability
    )

    test_auc = roc_auc_score(
        y_test,
        test_probability
    )

    print(
        f"\nValidation ROC-AUC: "
        f"{validation_auc:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_auc:.4f}"
    )

    # --------------------------------
    # Validation thresholds
    # --------------------------------

    validation_results = evaluate_thresholds(
        y_validation,
        validation_probability,
        "Validation",
        model_name
    )

    all_results.extend(
        validation_results
    )

    # --------------------------------
    # Test thresholds
    # --------------------------------

    test_results = evaluate_thresholds(
        y_test,
        test_probability,
        "Test",
        model_name
    )

    all_results.extend(
        test_results
    )


# ====================================
# Results DataFrame
# ====================================

results_df = pd.DataFrame(
    all_results
)


# ====================================
# Best validation threshold
# ====================================

validation_results_df = results_df[
    results_df["Dataset"] == "Validation"
].copy()


best_validation = (
    validation_results_df
    .sort_values(
        "F1",
        ascending=False
    )
    .groupby(
        "Model"
    )
    .first()
    .reset_index()
)


print("\n")
print("=" * 70)
print("BEST VALIDATION THRESHOLDS BY F1")
print("=" * 70)

print(
    best_validation[
        [
            "Model",
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Predicted_UP",
            "Predicted_DOWN",
        ]
    ].to_string(
        index=False
    )
)


# ====================================
# Apply validation-selected
# thresholds to test
# ====================================

print("\n")
print("=" * 70)
print("TEST PERFORMANCE USING VALIDATION THRESHOLD")
print("=" * 70)


for _, row in best_validation.iterrows():

    model_name = row["Model"]
    threshold = row["Threshold"]

    probabilities = test_probabilities[
        model_name
    ]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "-" * 60)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Validation threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Test Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Test Precision: {precision:.4f}"
    )

    print(
        f"Test Recall:    {recall:.4f}"
    )

    print(
        f"Test F1:        {f1:.4f}"
    )

    print(
        f"Test UP predictions: "
        f"{predictions.sum()}"
    )

    print(
        f"Test DOWN predictions: "
        f"{len(predictions) - predictions.sum()}"
    )


# ====================================
# Save results
# ====================================

output_file = (
    PROCESSED_DATA_DIR
    / "threshold_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ====================================
# Summary
# ====================================

print("\n")
print("=" * 70)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Results saved to: {output_file}"
)