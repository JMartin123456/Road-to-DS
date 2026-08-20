from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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
    roc_curve,
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

FIGURES_DIR = (
    BASE_DIR
    / "reports"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# Train models
# ====================================

validation_results = []
test_results = []

validation_probabilities = {}
test_probabilities = {}


for model_name, model in models.items():

    print("\n" + "=" * 40)
    print(f"Training: {model_name}")
    print("=" * 40)

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------
    # Validation
    # --------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    validation_probabilities[
        model_name
    ] = validation_probability

    validation_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_validation,
            validation_predictions
        ),
        "Precision": precision_score(
            y_validation,
            validation_predictions
        ),
        "Recall": recall_score(
            y_validation,
            validation_predictions
        ),
        "F1": f1_score(
            y_validation,
            validation_predictions
        ),
        "ROC_AUC": roc_auc_score(
            y_validation,
            validation_probability
        ),
    })

    # --------------------------------
    # Test
    # --------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    test_probabilities[
        model_name
    ] = test_probability

    test_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test,
            test_predictions
        ),
        "Precision": precision_score(
            y_test,
            test_predictions
        ),
        "Recall": recall_score(
            y_test,
            test_predictions
        ),
        "F1": f1_score(
            y_test,
            test_predictions
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            test_probability
        ),
    })

    # --------------------------------
    # Confusion matrix
    # --------------------------------

    print("\nValidation confusion matrix:")

    print(
        confusion_matrix(
            y_validation,
            validation_predictions
        )
    )

    print("\nTest confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )


# ====================================
# Results
# ====================================

validation_results_df = pd.DataFrame(
    validation_results
)

test_results_df = pd.DataFrame(
    test_results
)


print("\n")
print("=" * 60)
print("VALIDATION MODEL COMPARISON")
print("=" * 60)

print(
    validation_results_df.sort_values(
        "ROC_AUC",
        ascending=False
    ).to_string(
        index=False
    )
)


print("\n")
print("=" * 60)
print("TEST MODEL COMPARISON")
print("=" * 60)

print(
    test_results_df.sort_values(
        "ROC_AUC",
        ascending=False
    ).to_string(
        index=False
    )
)


# ====================================
# Best models
# ====================================

best_validation = validation_results_df.loc[
    validation_results_df["ROC_AUC"].idxmax()
]

best_test = test_results_df.loc[
    test_results_df["ROC_AUC"].idxmax()
]


print("\n")
print("=" * 60)
print("BEST VALIDATION MODEL")
print("=" * 60)

print(
    best_validation.to_string()
)


print("\n")
print("=" * 60)
print("BEST TEST MODEL")
print("=" * 60)

print(
    best_test.to_string()
)


# ====================================
# ROC curves - Validation
# ====================================

plt.figure(
    figsize=(8, 6)
)

for model_name, probabilities in validation_probabilities.items():

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_validation,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} (AUC={roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Validation"
)

plt.legend()

plt.tight_layout()

validation_roc_file = (
    FIGURES_DIR
    / "model_comparison_validation_roc.png"
)

plt.savefig(
    validation_roc_file,
    dpi=150
)

plt.close()


# ====================================
# ROC curves - Test
# ====================================

plt.figure(
    figsize=(8, 6)
)

for model_name, probabilities in test_probabilities.items():

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} (AUC={roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Test"
)

plt.legend()

plt.tight_layout()

test_roc_file = (
    FIGURES_DIR
    / "model_comparison_test_roc.png"
)

plt.savefig(
    test_roc_file,
    dpi=150
)

plt.close()


# ====================================
# Save results
# ====================================

validation_results_file = (
    PROCESSED_DATA_DIR
    / "model_comparison_validation.csv"
)

test_results_file = (
    PROCESSED_DATA_DIR
    / "model_comparison_test.csv"
)

validation_results_df.to_csv(
    validation_results_file,
    index=False
)

test_results_df.to_csv(
    test_results_file,
    index=False
)


# ====================================
# Summary
# ====================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON COMPLETED")
print("=" * 60)

print(
    f"Validation results saved to: "
    f"{validation_results_file}"
)

print(
    f"Test results saved to: "
    f"{test_results_file}"
)

print(
    f"Validation ROC figure saved to: "
    f"{validation_roc_file}"
)

print(
    f"Test ROC figure saved to: "
    f"{test_roc_file}"
)