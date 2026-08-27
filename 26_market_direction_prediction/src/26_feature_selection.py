"""Select base features using validation-only importance and model checks."""

import pandas as pd

from market_pipeline import BASE_FEATURES, PROCESSED_DIR, feature_selection


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    importance, results, _ = feature_selection(train, validation, test, BASE_FEATURES)
    importance = importance.rename(columns={
        "Importance_Mean": "Importance_RF",
        "GB_Importance": "Importance_GB",
    })
    importance.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)
    # Keep the historical one-row-per-feature-set layout and column names.
    results.to_csv(PROCESSED_DIR / "feature_selection_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
