from pathlib import Path

import numpy as np
import pandas as pd


# ====================================
# Config
# ====================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "eur_usd_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "eur_usd_features_advanced.csv"
)


# ====================================
# Load data
# ====================================

df = pd.read_csv(
    INPUT_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


print("\nOriginal shape:")
print(df.shape)


# ====================================
# 1. Momentum features
# ====================================

print("\nCreating momentum features...")

df["Momentum_3d"] = (
    df["Close"] / df["Close"].shift(3) - 1
)

df["Momentum_10d"] = (
    df["Close"] / df["Close"].shift(10) - 1
)

df["Momentum_30d"] = (
    df["Close"] / df["Close"].shift(30) - 1
)


# ====================================
# 2. Moving average relationships
# ====================================

print("Creating trend features...")

df["Price_vs_SMA_5"] = (
    df["Close"] / df["SMA_5"] - 1
)

df["Price_vs_SMA_50"] = (
    df["Close"] / df["SMA_50"] - 1
)

df["SMA_5_vs_SMA_20"] = (
    df["SMA_5"] / df["SMA_20"] - 1
)

df["SMA_20_vs_SMA_50"] = (
    df["SMA_20"] / df["SMA_50"] - 1
)


# ====================================
# 3. Trend direction
# ====================================

df["SMA_5_slope"] = (
    df["SMA_5"] / df["SMA_5"].shift(5) - 1
)

df["SMA_20_slope"] = (
    df["SMA_20"] / df["SMA_20"].shift(5) - 1
)

df["SMA_50_slope"] = (
    df["SMA_50"] / df["SMA_50"].shift(10) - 1
)


# ====================================
# 4. Volatility features
# ====================================

print("Creating volatility features...")

df["Volatility_5_vs_20"] = (
    df["Volatility_5"] /
    df["Volatility_20"]
)

df["Volatility_20_vs_50"] = (
    df["Volatility_20"] /
    df["Volatility_50"]
)

df["Volatility_5_change"] = (
    df["Volatility_5"] /
    df["Volatility_5"].shift(5) - 1
)

df["Volatility_20_change"] = (
    df["Volatility_20"] /
    df["Volatility_20"].shift(10) - 1
)


# ====================================
# 5. Price range features
# ====================================

print("Creating price range features...")

df["Range_5d"] = (
    df["High"].rolling(5).max()
    /
    df["Low"].rolling(5).min()
    - 1
)

df["Range_20d"] = (
    df["High"].rolling(20).max()
    /
    df["Low"].rolling(20).min()
    - 1
)

df["Range_50d"] = (
    df["High"].rolling(50).max()
    /
    df["Low"].rolling(50).min()
    - 1
)


# ====================================
# 6. Price position inside recent range
# ====================================

print("Creating price position features...")

high_20 = df["High"].rolling(20).max()
low_20 = df["Low"].rolling(20).min()

high_50 = df["High"].rolling(50).max()
low_50 = df["Low"].rolling(50).min()

df["Price_Position_20"] = (
    (df["Close"] - low_20)
    /
    (high_20 - low_20)
)

df["Price_Position_50"] = (
    (df["Close"] - low_50)
    /
    (high_50 - low_50)
)


# ====================================
# 7. Candle body / wick relationships
# ====================================

print("Creating candle structure features...")

df["Body_to_Range"] = (
    df["Body_Size_Pct"]
    /
    df["High_Low_Range_Pct"]
)

df["Upper_Wick_to_Range"] = (
    df["Upper_Wick_Pct"]
    /
    df["High_Low_Range_Pct"]
)

df["Lower_Wick_to_Range"] = (
    df["Lower_Wick_Pct"]
    /
    df["High_Low_Range_Pct"]
)


# ====================================
# 8. Candle direction
# ====================================

df["Candle_Direction"] = np.where(
    df["Close"] > df["Open"],
    1,
    0
)


# ====================================
# 9. Consecutive candle direction
# ====================================

df["Previous_Candle_Direction"] = (
    df["Candle_Direction"].shift(1)
)

df["Previous_Return"] = (
    df["Return_1d"].shift(1)
)


# ====================================
# 10. Momentum acceleration
# ====================================

print("Creating momentum acceleration features...")

df["Momentum_Acceleration"] = (
    df["Return_1d"]
    - df["Return_1d"].shift(1)
)

df["Return_5d_change"] = (
    df["Return_5d"]
    - df["Return_5d"].shift(5)
)


# ====================================
# 11. Rolling return statistics
# ====================================

print("Creating rolling statistics...")

df["Return_Mean_5"] = (
    df["Return_1d"]
    .rolling(5)
    .mean()
)

df["Return_Mean_20"] = (
    df["Return_1d"]
    .rolling(20)
    .mean()
)

df["Return_Std_5"] = (
    df["Return_1d"]
    .rolling(5)
    .std()
)

df["Return_Std_20"] = (
    df["Return_1d"]
    .rolling(20)
    .std()
)


# ====================================
# 12. Positive / negative return ratio
# ====================================

df["Positive_Return_Ratio_10"] = (
    df["Return_1d"]
    .rolling(10)
    .apply(
        lambda x: np.mean(x > 0),
        raw=True
    )
)

df["Positive_Return_Ratio_20"] = (
    df["Return_1d"]
    .rolling(20)
    .apply(
        lambda x: np.mean(x > 0),
        raw=True
    )
)


# ====================================
# 13. Distance from recent high / low
# ====================================

recent_high_20 = (
    df["High"]
    .rolling(20)
    .max()
)

recent_low_20 = (
    df["Low"]
    .rolling(20)
    .min()
)

df["Distance_From_High_20"] = (
    df["Close"] / recent_high_20 - 1
)

df["Distance_From_Low_20"] = (
    df["Close"] / recent_low_20 - 1
)


# ====================================
# 14. Volatility-adjusted momentum
# ====================================

df["Momentum_5_Vol_Adjusted"] = (
    df["Return_5d"]
    /
    df["Volatility_20"]
)

df["Momentum_20_Vol_Adjusted"] = (
    df["Return_20d"]
    /
    df["Volatility_50"]
)


# ====================================
# 15. Clean infinite values
# ====================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ====================================
# Overview
# ====================================

print("\nAdvanced feature engineering completed.")

print("\nNew shape:")
print(df.shape)

print("\nNew features:")

original_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "Volatility_20",
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "Price_vs_SMA_20",
    "Target",
    "Volatility_5",
    "Volatility_50",
    "Volatility_Ratio_5_20",
    "High_Low_Range_Pct",
    "Open_Close_Range_Pct",
    "Body_Size_Pct",
    "Upper_Wick",
    "Lower_Wick",
    "Upper_Wick_Pct",
    "Lower_Wick_Pct",
]

new_features = [
    column
    for column in df.columns
    if column not in original_columns
]

print(new_features)

print("\nNumber of new features:")
print(len(new_features))


# ====================================
# Missing values
# ====================================

print("\nMissing values in new features:")

print(
    df[new_features]
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)


# ====================================
# Save
# ====================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)