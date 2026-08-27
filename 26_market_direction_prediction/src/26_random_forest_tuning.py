"""Tune the base random-forest model on chronological validation data."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from market_pipeline import BASE_FEATURES, PROCESSED_DIR, tune_model


PARAMETERS = [
    {"n_estimators": 200, "max_depth": depth, "min_samples_split": split, "min_samples_leaf": leaf}
    for depth, split, leaf in ((3, 2, 1), (5, 2, 1), (8, 2, 1), (5, 5, 2), (8, 5, 2), (10, 10, 4))
]


def main() -> None:
    train, validation, test = (
        pd.read_csv(PROCESSED_DIR / f"{name}.csv") for name in ("train", "validation", "test")
    )
    results, _, parameters, final = tune_model(
        RandomForestClassifier, PARAMETERS, train, validation, test, BASE_FEATURES, "Random Forest"
    )
    results.to_csv(PROCESSED_DIR / "random_forest_tuning.csv", index=False)
    print("Best parameters:", parameters)
    print("Validation:", final["Validation"])
    print("Test:", final["Test"])


if __name__ == "__main__":
    main()
