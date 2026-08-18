from pathlib import Path

import requests
import pandas as pd


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

print(df.head())
print(df.tail())
print(df.shape)