"""Tune both advanced tree models and save the historical result tables."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from market_pipeline import ADVANCED_FEATURES, PROCESSED_DIR, tune_model


RF_PARAMETERS = [
    {"n_estimators": n, "max_depth": depth, "min_samples_split": split,
     "min_samples_leaf": leaf, "max_features": "sqrt"}
    for n, depth, split, leaf in (
        (200, 3, 2, 1), (300, 3, 2, 1), (300, 5, 2, 1),
        (300, 5, 5, 2), (500, 3, 5, 2), (500, 5, 5, 2),
    )
]
GB_PARAMETERS = [
    {"n_estimators": n, "learning_rate": rate, "max_depth": depth,
     "min_samples_split": split, "min_samples_leaf": leaf}
    for n, rate, depth, split, leaf in (
        (100, 0.01, 2, 2, 1), (200, 0.03, 2, 2, 1),
        (300, 0.03, 3, 5, 2), (200, 0.05, 2, 2, 1),
        (300, 0.05, 3, 5, 2),
    )
]


def main() -> None:
    train, validation, test = (
        pd.read_csv(PROCESSED_DIR / f"{name}_advanced.csv")
        for name in ("train", "validation", "test")
    )
    rf_results, _, rf_parameters, rf_final = tune_model(
        RandomForestClassifier, RF_PARAMETERS, train, validation, test, ADVANCED_FEATURES, "Random Forest"
    )
    gb_results, _, gb_parameters, gb_final = tune_model(
        GradientBoostingClassifier, GB_PARAMETERS, train, validation, test, ADVANCED_FEATURES, "Gradient Boosting"
    )
    rf_results.to_csv(PROCESSED_DIR / "advanced_random_forest_tuning.csv", index=False)
    gb_results.to_csv(PROCESSED_DIR / "advanced_gradient_boosting_tuning.csv", index=False)
    comparison = pd.DataFrame([
        {"Model": "Random Forest", **{f"Validation_{k}": v for k, v in rf_final["Validation"].items()},
         **{f"Test_{k}": v for k, v in rf_final["Test"].items()}},
        {"Model": "Gradient Boosting", **{f"Validation_{k}": v for k, v in gb_final["Validation"].items()},
         **{f"Test_{k}": v for k, v in gb_final["Test"].items()}},
    ])
    comparison.to_csv(PROCESSED_DIR / "advanced_tuning_model_comparison.csv", index=False)
    print("Best RF:", rf_parameters, rf_final)
    print("Best GB:", gb_parameters, gb_final)


if __name__ == "__main__":
    main()
