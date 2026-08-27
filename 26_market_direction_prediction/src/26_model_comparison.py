"""Compare the three base models and save tables/ROC figures."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from market_pipeline import BASE_FEATURES, FIGURES_DIR, PROCESSED_DIR, compare_models


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    validation_result, test_result, probabilities, _ = compare_models(
        train, validation, test, BASE_FEATURES
    )
    validation_result.to_csv(PROCESSED_DIR / "model_comparison_validation.csv", index=False)
    test_result.to_csv(PROCESSED_DIR / "model_comparison_test.csv", index=False)
    for dataset, frame, output in (
        ("Validation", validation, "model_comparison_validation_roc.png"),
        ("Test", test, "model_comparison_test_roc.png"),
    ):
        plt.figure(figsize=(8, 6))
        for name, probability in probabilities[dataset].items():
            fpr, tpr, _ = roc_curve(frame["Target"], probability)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(frame['Target'], probability):.3f})")
        plt.plot([0, 1], [0, 1], "--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / output, dpi=150)
        plt.close()
    print(validation_result.to_string(index=False))
    print(test_result.to_string(index=False))


if __name__ == "__main__":
    main()
