"""Select advanced features using validation-only importance and model checks."""

import pandas as pd

from market_pipeline import ADVANCED_FEATURES, PROCESSED_DIR, feature_selection


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train_advanced.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation_advanced.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_advanced.csv")
    importance, results, selected = feature_selection(
        train, validation, test, ADVANCED_FEATURES, advanced=True
    )
    importance.to_csv(PROCESSED_DIR / "advanced_feature_importance.csv", index=False)
    results.to_csv(PROCESSED_DIR / "advanced_feature_selection_results.csv", index=False)
    pd.DataFrame({
        "RF_Best_Feature_Set": pd.Series(selected["RF"]),
        "GB_Best_Feature_Set": pd.Series(selected["GB"]),
    }).to_csv(PROCESSED_DIR / "advanced_selected_features.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
