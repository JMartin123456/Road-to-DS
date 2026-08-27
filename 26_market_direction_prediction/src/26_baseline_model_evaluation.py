"""Evaluate the baseline and retain its historical diagnostic figures."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from market_pipeline import BASE_FEATURES, FIGURES_DIR, PROCESSED_DIR, metrics, model_factories


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    model = model_factories()["Logistic Regression"].fit(train[BASE_FEATURES], train["Target"])
    predictions = {}
    probabilities = {}
    for name, frame in (("validation", validation), ("test", test)):
        probabilities[name] = model.predict_proba(frame[BASE_FEATURES])[:, 1]
        predictions[name] = (probabilities[name] >= 0.5).astype(int)
        print(name.title(), metrics(frame["Target"], predictions[name], probabilities[name]))
        ConfusionMatrixDisplay.from_predictions(frame["Target"], predictions[name])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"baseline_confusion_matrix_{name}.png", dpi=150)
        plt.close()
        RocCurveDisplay.from_predictions(frame["Target"], probabilities[name])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"baseline_roc_{name}.png", dpi=150)
        plt.close()
    plt.figure(figsize=(8, 5))
    plt.hist(probabilities["validation"], bins=30)
    plt.xlabel("Predicted probability of Target = 1")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_probability_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
