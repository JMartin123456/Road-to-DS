"""Evaluate decision thresholds without using the test set for selection."""

import pandas as pd

from market_pipeline import ADVANCED_FEATURES, PROCESSED_DIR, compare_models, evaluate_thresholds


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train_advanced.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation_advanced.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_advanced.csv")
    _, _, probabilities, _ = compare_models(train, validation, test, ADVANCED_FEATURES)
    tables = [
        evaluate_thresholds(validation["Target"], probabilities["Validation"][name], name, "Validation")
        for name in probabilities["Validation"]
    ]
    tables.extend(
        evaluate_thresholds(test["Target"], probabilities["Test"][name], name, "Test")
        for name in probabilities["Test"]
    )
    results = pd.concat(tables, ignore_index=True)
    results.to_csv(PROCESSED_DIR / "threshold_analysis.csv", index=False)
    best = (results[results["Dataset"] == "Validation"]
            .sort_values(["Model", "F1", "Threshold"], ascending=[True, False, True])
            .groupby("Model", as_index=False).first())
    print(best[["Model", "Threshold", "F1"]].to_string(index=False))


if __name__ == "__main__":
    main()
