from pathlib import Path

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

PROCESSED_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ====================================
# Load raw data
# ====================================

input_file = (
    RAW_DATA_DIR
    / "eur_usd_daily.csv"
)

df = pd.read_csv(
    input_file
)


# ====================================
# Data types
# ====================================

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)


# ====================================
# Returns
# ====================================

df["Return_1d"] = (
    df["Close"].pct_change(1)
)

df["Return_5d"] = (
    df["Close"].pct_change(5)
)

df["Return_20d"] = (
    df["Close"].pct_change(20)
)


# ====================================
# Rolling Volatility
# ====================================

df["Volatility_20"] = (
    df["Return_1d"]
    .rolling(20)
    .std()
)


# ====================================
# Trend Features
# Simple Moving Averages
# ====================================

df["SMA_5"] = (
    df["Close"]
    .rolling(5)
    .mean()
)

df["SMA_20"] = (
    df["Close"]
    .rolling(20)
    .mean()
)

df["SMA_50"] = (
    df["Close"]
    .rolling(50)
    .mean()
)

df["Price_vs_SMA_20"] = (
    df["Close"] / df["SMA_20"] - 1
)


# ====================================
# Target
# ====================================

future_close = (
    df["Close"].shift(-1)
)

df["Target"] = (
    future_close > df["Close"]
).astype("Int64")

# ====================================
# Volatility Features
# ====================================

df["Volatility_5"] = (
    df["Return_1d"]
    .rolling(5)
    .std()
)

df["Volatility_20"] = (
    df["Return_1d"]
    .rolling(20)
    .std()
)

df["Volatility_50"] = (
    df["Return_1d"]
    .rolling(50)
    .std()
)

df["Volatility_Ratio_5_20"] = (
    df["Volatility_5"]
    / df["Volatility_20"]
)

print("\nVolatility features:")

print(
    df[
        [
            "Date",
            "Return_1d",
            "Volatility_5",
            "Volatility_20",
            "Volatility_50",
            "Volatility_Ratio_5_20",
            "Target"
        ]
    ].tail(10)
)

print("\nVolatility statistics:")

print(
    df[
        [
            "Volatility_5",
            "Volatility_20",
            "Volatility_50",
            "Volatility_Ratio_5_20"
        ]
    ].describe()
)


# ====================================
# Summary
# ====================================

print("\nTrend features:")

print(
    df[
        [
            "Date",
            "Close",
            "SMA_5",
            "SMA_20",
            "SMA_50",
            "Price_vs_SMA_20",
            "Target"
        ]
    ].tail(10)
)

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)


# ====================================
# Save processed data
# ====================================

output_file = (
    PROCESSED_DATA_DIR
    / "eur_usd_features.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nFeature engineering completed.")

print(f"Saved to: {output_file}")