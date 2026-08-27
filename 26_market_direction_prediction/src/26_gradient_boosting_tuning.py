"""Tune the base gradient-boosting model on chronological validation data."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from market_pipeline import BASE_FEATURES, PROCESSED_DIR, tune_model


PARAMETERS = [
    {"n_estimators": n, "learning_rate": rate, "max_depth": depth,
     "min_samples_split": split, "min_samples_leaf": leaf}
    for n, rate, depth, split, leaf in (
        (100, 0.01, 2, 2, 1), (200, 0.03, 2, 2, 1), (300, 0.03, 3, 5, 2),
        (200, 0.05, 2, 2, 1), (300, 0.05, 3, 5, 2),
    )
]


def main() -> None:
    train, validation, test = (
        pd.read_csv(PROCESSED_DIR / f"{name}.csv") for name in ("train", "validation", "test")
    )
    results, _, parameters, final = tune_model(
        GradientBoostingClassifier, PARAMETERS, train, validation, test, BASE_FEATURES, "Gradient Boosting"
    )
    results.to_csv(PROCESSED_DIR / "gradient_boosting_tuning.csv", index=False)
    print("Best parameters:", parameters)
    print("Validation:", final["Validation"])
    print("Test:", final["Test"])


if __name__ == "__main__":
    main()
