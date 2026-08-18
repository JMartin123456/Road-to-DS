from pathlib import Path

import requests
import pandas as pd


# ====================================
# Config
# ====================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = (
    BASE_DIR
    / "data"
    / "raw"
)

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ====================================
# API
# ====================================

API_KEY = "YOUR_API_KEY"

url = (
    "https://www.alphavantage.co/query"
    "?function=FX_DAILY"
    "&from_symbol=EUR"
    "&to_symbol=USD"
    "&outputsize=full"
    f"&apikey={API_KEY}"
)


# ====================================
# Load data
# ====================================

response = requests.get(
    url,
    timeout=30
)

response.raise_for_status()

data = response.json()


# ====================================
# Validate API response
# ====================================

if "Time Series FX (Daily)" not in data:

    print("\nAlpha Vantage API error:")

    for key, value in data.items():
        print(f"{key}: {value}")

    raise RuntimeError(
        "Failed to load EUR/USD daily data."
    )


# ====================================
# DataFrame
# ====================================

df = pd.DataFrame.from_dict(
    data["Time Series FX (Daily)"],
    orient="index"
)

df = df.reset_index()

df.columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close"
]


# ====================================
# Data types
# ====================================

df["Date"] = pd.to_datetime(
    df["Date"]
)

df[
    [
        "Open",
        "High",
        "Low",
        "Close"
    ]
] = df[
    [
        "Open",
        "High",
        "Low",
        "Close"
    ]
].astype(float)


# ====================================
# Sort
# ====================================

df = df.sort_values(
    "Date"
)


# ====================================
# Save
# ====================================

file_path = (
    RAW_DATA_DIR
    / "eur_usd_daily.csv"
)

df.to_csv(
    file_path,
    index=False
)


# ====================================
# Summary
# ====================================

print("Data collection completed.")

print(f"Saved to: {file_path}")

print(f"Rows: {len(df)}")

print("\nFirst rows:")
print(df.head())

print("\nLast rows:")
print(df.tail())