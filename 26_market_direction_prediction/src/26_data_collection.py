"""Collect EUR/USD daily prices.

Set ``ALPHAVANTAGE_API_KEY`` before running this script.
"""

from market_pipeline import collect_data


if __name__ == "__main__":
    frame = collect_data()
    print(f"Collected {len(frame):,} rows.")
