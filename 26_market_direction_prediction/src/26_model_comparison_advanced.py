"""Compare models using the advanced feature set."""

import pandas as pd

from market_pipeline import ADVANCED_FEATURES, PROCESSED_DIR, compare_models


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train_advanced.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation_advanced.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_advanced.csv")
    validation_result, test_result, _, _ = compare_models(
        train, validation, test, ADVANCED_FEATURES
    )
    validation_result.to_csv(PROCESSED_DIR / "model_comparison_advanced_validation.csv", index=False)
    test_result.to_csv(PROCESSED_DIR / "model_comparison_advanced_test.csv", index=False)
    print(validation_result.to_string(index=False))
    print(test_result.to_string(index=False))


if __name__ == "__main__":
    main()
