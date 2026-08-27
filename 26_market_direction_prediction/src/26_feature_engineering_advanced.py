"""Build advanced features from the base feature data set."""

from market_pipeline import PROCESSED_DIR, engineer_advanced_features
import pandas as pd


def main() -> None:
    source = PROCESSED_DIR / "eur_usd_features.csv"
    frame = engineer_advanced_features(pd.read_csv(source))
    output = PROCESSED_DIR / "eur_usd_features_advanced.csv"
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} rows and {len(frame.columns):,} columns to {output}")


if __name__ == "__main__":
    main()
