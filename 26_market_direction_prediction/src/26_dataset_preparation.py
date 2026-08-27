"""Clean and chronologically split the base feature data."""

import pandas as pd

from market_pipeline import BASE_FEATURES, PROCESSED_DIR, prepare_datasets


def main() -> None:
    frame = pd.read_csv(PROCESSED_DIR / "eur_usd_features.csv")
    train, validation, test, clean = prepare_datasets(frame, BASE_FEATURES)
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    clean.to_csv(PROCESSED_DIR / "eur_usd_model_data.csv", index=False)
    print(f"Prepared {len(clean):,} usable rows: {len(train):,}/{len(validation):,}/{len(test):,}")


if __name__ == "__main__":
    main()
