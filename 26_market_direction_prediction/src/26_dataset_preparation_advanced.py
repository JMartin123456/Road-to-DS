"""Clean and chronologically split the advanced feature data."""

import pandas as pd

from market_pipeline import ADVANCED_FEATURES, PROCESSED_DIR, prepare_datasets


def main() -> None:
    frame = pd.read_csv(PROCESSED_DIR / "eur_usd_features_advanced.csv")
    train, validation, test, clean = prepare_datasets(frame, ADVANCED_FEATURES)
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_csv(PROCESSED_DIR / f"{name}_advanced.csv", index=False)
    clean.to_csv(PROCESSED_DIR / "eur_usd_model_data_advanced.csv", index=False)
    print(f"Prepared {len(clean):,} usable rows: {len(train):,}/{len(validation):,}/{len(test):,}")


if __name__ == "__main__":
    main()
