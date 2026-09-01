from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "Return_1d", "Return_5d", "Return_20d", "Volatility_5",
    "Volatility_20", "Volatility_50", "Volatility_Ratio_5_20",
    "SMA_5", "SMA_20", "SMA_50", "Price_vs_SMA_20",
    "High_Low_Range_Pct", "Open_Close_Range_Pct", "Body_Size_Pct",
    "Upper_Wick_Pct", "Lower_Wick_Pct",
]

ADVANCED_FEATURES = [
    *BASE_FEATURES,
    "Upper_Wick", "Lower_Wick",
    "Momentum_3d", "Momentum_10d", "Momentum_30d",
    "Momentum_Acceleration", "Return_5d_change",
    "Price_vs_SMA_5", "Price_vs_SMA_50", "SMA_5_vs_SMA_20",
    "SMA_20_vs_SMA_50", "SMA_5_slope", "SMA_20_slope", "SMA_50_slope",
    "Volatility_5_vs_20", "Volatility_20_vs_50", "Volatility_5_change",
    "Volatility_20_change", "Range_5d", "Range_20d", "Range_50d",
    "Price_Position_20", "Price_Position_50", "Distance_From_High_20",
    "Distance_From_Low_20", "Body_to_Range", "Upper_Wick_to_Range",
    "Lower_Wick_to_Range", "Candle_Direction",
    "Previous_Candle_Direction", "Previous_Return", "Return_Mean_5",
    "Return_Mean_20", "Return_Std_5", "Return_Std_20",
    "Positive_Return_Ratio_10", "Positive_Return_Ratio_20",
    "Momentum_5_Vol_Adjusted", "Momentum_20_Vol_Adjusted",
]

def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)

def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create the original feature set and a next-day direction target."""
    frame = raw.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.sort_values("Date").reset_index(drop=True)
    close = frame["Close"]
    frame["Return_1d"] = close.pct_change()
    frame["Return_5d"] = close.pct_change(5)
    frame["Return_20d"] = close.pct_change(20)
    frame["Volatility_5"] = frame["Return_1d"].rolling(5, min_periods=5).std()
    frame["Volatility_20"] = frame["Return_1d"].rolling(20, min_periods=20).std()
    frame["Volatility_50"] = frame["Return_1d"].rolling(50, min_periods=50).std()
    frame["Volatility_Ratio_5_20"] = _safe_ratio(
        frame["Volatility_5"], frame["Volatility_20"]
    )
    for window in (5, 20, 50):
        frame[f"SMA_{window}"] = close.rolling(window, min_periods=window).mean()
    frame["Price_vs_SMA_20"] = _safe_ratio(close, frame["SMA_20"]) - 1
    frame["High_Low_Range_Pct"] = (frame["High"] - frame["Low"]) / close
    frame["Open_Close_Range_Pct"] = (close - frame["Open"]) / frame["Open"]
    frame["Body_Size_Pct"] = (close - frame["Open"]).abs() / frame["Open"]
    frame["Upper_Wick"] = frame["High"] - frame[["Open", "Close"]].max(axis=1)
    frame["Lower_Wick"] = frame[["Open", "Close"]].min(axis=1) - frame["Low"]
    frame["Upper_Wick_Pct"] = frame["Upper_Wick"] / close
    frame["Lower_Wick_Pct"] = frame["Lower_Wick"] / close
    future_close = close.shift(-1)
    frame["Target"] = (future_close > close).astype("float")
    frame.loc[future_close.isna(), "Target"] = np.nan
    return frame.replace([np.inf, -np.inf], np.nan)


def engineer_advanced_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add advanced features without changing the base columns."""
    frame = features.copy().sort_values("Date").reset_index(drop=True)
    close = frame["Close"]
    frame["Momentum_3d"] = close.pct_change(3)
    frame["Momentum_10d"] = close.pct_change(10)
    frame["Momentum_30d"] = close.pct_change(30)
    frame["Price_vs_SMA_5"] = _safe_ratio(close, frame["SMA_5"]) - 1
    frame["Price_vs_SMA_50"] = _safe_ratio(close, frame["SMA_50"]) - 1
    frame["SMA_5_vs_SMA_20"] = _safe_ratio(frame["SMA_5"], frame["SMA_20"]) - 1
    frame["SMA_20_vs_SMA_50"] = _safe_ratio(frame["SMA_20"], frame["SMA_50"]) - 1
    frame["SMA_5_slope"] = _safe_ratio(frame["SMA_5"], frame["SMA_5"].shift(5)) - 1
    frame["SMA_20_slope"] = _safe_ratio(frame["SMA_20"], frame["SMA_20"].shift(5)) - 1
    frame["SMA_50_slope"] = _safe_ratio(frame["SMA_50"], frame["SMA_50"].shift(10)) - 1
    frame["Volatility_5_vs_20"] = _safe_ratio(frame["Volatility_5"], frame["Volatility_20"])
    frame["Volatility_20_vs_50"] = _safe_ratio(frame["Volatility_20"], frame["Volatility_50"])
    frame["Volatility_5_change"] = _safe_ratio(
        frame["Volatility_5"], frame["Volatility_5"].shift(5)
    ) - 1
    frame["Volatility_20_change"] = _safe_ratio(
        frame["Volatility_20"], frame["Volatility_20"].shift(10)
    ) - 1
    for window in (5, 20, 50):
        high = frame["High"].rolling(window, min_periods=window).max()
        low = frame["Low"].rolling(window, min_periods=window).min()
        frame[f"Range_{window}d"] = _safe_ratio(high, low) - 1
    high20 = frame["High"].rolling(20, min_periods=20).max()
    low20 = frame["Low"].rolling(20, min_periods=20).min()
    high50 = frame["High"].rolling(50, min_periods=50).max()
    low50 = frame["Low"].rolling(50, min_periods=50).min()
    frame["Price_Position_20"] = _safe_ratio(close - low20, high20 - low20)
    frame["Price_Position_50"] = _safe_ratio(close - low50, high50 - low50)
    frame["Distance_From_High_20"] = _safe_ratio(close, high20) - 1
    frame["Distance_From_Low_20"] = _safe_ratio(close, low20) - 1
    frame["Body_to_Range"] = _safe_ratio(frame["Body_Size_Pct"], frame["High_Low_Range_Pct"])
    frame["Upper_Wick_to_Range"] = _safe_ratio(
        frame["Upper_Wick_Pct"], frame["High_Low_Range_Pct"]
    )
    frame["Lower_Wick_to_Range"] = _safe_ratio(
        frame["Lower_Wick_Pct"], frame["High_Low_Range_Pct"]
    )
    frame["Candle_Direction"] = (frame["Close"] > frame["Open"]).astype("float")
    frame["Previous_Candle_Direction"] = frame["Candle_Direction"].shift(1)
    frame["Previous_Return"] = frame["Return_1d"].shift(1)
    frame["Momentum_Acceleration"] = frame["Return_1d"] - frame["Return_1d"].shift(1)
    frame["Return_5d_change"] = frame["Return_5d"] - frame["Return_5d"].shift(5)
    frame["Return_Mean_5"] = frame["Return_1d"].rolling(5, min_periods=5).mean()
    frame["Return_Mean_20"] = frame["Return_1d"].rolling(20, min_periods=20).mean()
    frame["Return_Std_5"] = frame["Return_1d"].rolling(5, min_periods=5).std()
    frame["Return_Std_20"] = frame["Return_1d"].rolling(20, min_periods=20).std()
    frame["Positive_Return_Ratio_10"] = frame["Return_1d"].rolling(
        10, min_periods=10
    ).apply(lambda values: np.mean(values > 0), raw=True)
    frame["Positive_Return_Ratio_20"] = frame["Return_1d"].rolling(
        20, min_periods=20
    ).apply(lambda values: np.mean(values > 0), raw=True)
    frame["Momentum_5_Vol_Adjusted"] = _safe_ratio(
        frame["Return_5d"], frame["Volatility_20"]
    )
    frame["Momentum_20_Vol_Adjusted"] = _safe_ratio(
        frame["Return_20d"], frame["Volatility_50"]
    )
    return frame.replace([np.inf, -np.inf], np.nan)