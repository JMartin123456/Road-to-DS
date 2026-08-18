from pathlib import Path

import requests
import pandas as pd

import matplotlib.pyplot as plt

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

response = requests.get(url)

data = response.json()

#print(data)


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
# Data Validation
# ====================================

# print("\nMissing values:")
# print(df.isna().sum())

# print("\nDuplicate dates:")
# print(df["Date"].duplicated().sum())

# print("\nData types:")
# print(df.dtypes)

# print("\nDescriptive statistics:")
# print(df.describe())

# invalid_high = (
#     (df["High"] < df["Open"]) |
#     (df["High"] < df["Close"])
# ).sum()

# invalid_low = (
#     (df["Low"] > df["Open"]) |
#     (df["Low"] > df["Close"])
# ).sum()

# print("\nOHLC validation:")
# print("Invalid High values:", invalid_high)
# print("Invalid Low values:", invalid_low)

# ====================================
# Returns
# ====================================

df["Return"] = df["Close"].pct_change()

print("\nReturns:")
print(df["Return"].describe())

print("\nLargest positive returns:")
print(
    df.nlargest(10, "Return")[
        ["Date", "Close", "Return"]
    ]
)

print("\nLargest negative returns:")
print(
    df.nsmallest(10, "Return")[
        ["Date", "Close", "Return"]
    ]
)

# ====================================
# Rolling Volatility
# ====================================

df["Volatility_20"] = (
    df["Return"]
    .rolling(20)
    .std()
)

print("\n20-day Rolling Volatility:")
print(df["Volatility_20"].describe())

# ====================================
# Volatility chart
# ====================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Volatility_20"]
)

plt.title("EUR/USD 20-Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")

plt.tight_layout()
plt.show()

# ====================================
# Return Distribution
# ====================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["Return"].dropna(),
    bins=100
)

plt.title("Distribution of EUR/USD Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ====================================
# Save
# ====================================

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "eur_usd_daily.csv"
)

df.to_csv(
    file_path,
    index=False
)

# print(df.head())
# print(df.tail())
# print(df.shape)