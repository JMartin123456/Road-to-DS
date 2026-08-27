"""Build the base EUR/USD feature data set."""

from market_pipeline import PROCESSED_DIR, RAW_DIR, engineer_features, load_ohlc


def main() -> None:
    frame = engineer_features(load_ohlc(RAW_DIR / "eur_usd_daily.csv"))
    output = PROCESSED_DIR / "eur_usd_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} rows to {output}")


if __name__ == "__main__":
    main()
