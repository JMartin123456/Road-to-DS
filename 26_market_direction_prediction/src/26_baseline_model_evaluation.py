from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
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
# Naive baseline
# Always predict the most common class
# ====================================

majority_class = (
    y_train
    .value_counts()
    .idxmax()
)

print("\nNaive baseline:")
print(
    "Majority class:",
    majority_class
)


naive_validation_predictions = (
    [majority_class] * len(y_validation)
)

naive_test_predictions = (
    [majority_class] * len(y_test)
)


print(
    "\nNaive Validation Accuracy:",
    f"{accuracy_score(y_validation, naive_validation_predictions):.4f}"
)

print(
    "Naive Test Accuracy:",
    f"{accuracy_score(y_test, naive_test_predictions):.4f}"
)


# ====================================
# Logistic Regression
# ====================================

model = LogisticRegression(
    max_iter=1000
)

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
# Prediction distribution
# ====================================

print("\nLogistic Regression predictions:")

print(
    "\nValidation:"
)

print(
    pd.Series(
        validation_predictions
    ).value_counts()
)


print(
    "\nTest:"
)

print(
    pd.Series(
        test_predictions
    ).value_counts()
)


# ====================================
# Evaluation
# ====================================

def print_metrics(
    y_true,
    predictions,
    probabilities,
    dataset_name
):

    print(
        f"\n{'=' * 40}"
    )

    print(
        dataset_name
    )

    print(
        f"{'=' * 40}"
    )

    print(
        "Accuracy:",
        f"{accuracy_score(y_true, predictions):.4f}"
    )

    print(
        "Precision:",
        f"{precision_score(y_true, predictions):.4f}"
    )

    print(
        "Recall:",
        f"{recall_score(y_true, predictions):.4f}"
    )

    print(
        "F1:",
        f"{f1_score(y_true, predictions):.4f}"
    )

    print(
        "ROC-AUC:",
        f"{roc_auc_score(y_true, probabilities):.4f}"
    )


print_metrics(
    y_validation,
    validation_predictions,
    validation_probabilities,
    "Validation"
)

print_metrics(
    y_test,
    test_predictions,
    test_probabilities,
    "Test"
)


# ====================================
# Confusion Matrix - Validation
# ====================================

cm_validation = confusion_matrix(
    y_validation,
    validation_predictions
)

print("\nValidation confusion matrix:")

print(
    cm_validation
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_validation,
    display_labels=[0, 1]
)

disp.plot()

plt.title(
    "Logistic Regression - Validation Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "baseline_confusion_matrix_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ====================================
# Confusion Matrix - Test
# ====================================

cm_test = confusion_matrix(
    y_test,
    test_predictions
)

print("\nTest confusion matrix:")

print(
    cm_test
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_test,
    display_labels=[0, 1]
)

disp.plot()

plt.title(
    "Logistic Regression - Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "baseline_confusion_matrix_test.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ====================================
# ROC Curve - Validation
# ====================================

RocCurveDisplay.from_predictions(
    y_validation,
    validation_probabilities
)

plt.title(
    "Logistic Regression - Validation ROC Curve"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "baseline_roc_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ====================================
# ROC Curve - Test
# ====================================

RocCurveDisplay.from_predictions(
    y_test,
    test_probabilities
)

plt.title(
    "Logistic Regression - Test ROC Curve"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "baseline_roc_test.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ====================================
# Prediction probabilities
# ====================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    validation_probabilities,
    bins=30
)

plt.title(
    "Baseline Prediction Probabilities - Validation"
)

plt.xlabel(
    "Predicted probability of Target = 1"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR
    / "baseline_probability_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ====================================
# Summary
# ====================================

print(
    "\nBaseline evaluation completed."
)

print(
    "Figures saved to:",
    FIGURES_DIR
)