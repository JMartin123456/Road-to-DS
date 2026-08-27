"""Generate feature, distribution, effect-size, and model diagnostics."""

import pandas as pd

from market_pipeline import ADVANCED_FEATURES, PROCESSED_DIR, diagnostics, save_tables


def main() -> None:
    frames = [pd.read_csv(PROCESSED_DIR / f"{name}_advanced.csv")
              for name in ("train", "validation", "test")]
    tables = diagnostics(*frames, ADVANCED_FEATURES)
    save_tables(tables)
    print(f"Saved {len(tables)} diagnostic tables.")


if __name__ == "__main__":
    main()
