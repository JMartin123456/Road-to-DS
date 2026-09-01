"""Data collection and validation for the EUR/USD prediction project."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
RANDOM_STATE = 42


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _validate_ohlc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an OHLC frame without writing intermediate files."""
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")

    numeric = ["Open", "High", "Low", "Close"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")

    if frame["Date"].duplicated().any():
        raise ValueError("Input contains duplicate dates.")

    if (frame[numeric] <= 0).any().any():
        raise ValueError("OHLC values must be positive.")

    if (frame["High"] < frame[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("High must be at least Open and Close.")

    if (frame["Low"] > frame[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("Low must be at most Open and Close.")

    return frame.sort_values("Date").reset_index(drop=True)


def load_ohlc(path: Path) -> pd.DataFrame:
    """Load and validate a chronological OHLC data set."""
    return _validate_ohlc_frame(pd.read_csv(path))


def load_ohlc_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an already materialized OHLC frame."""
    return _validate_ohlc_frame(frame)


def collect_data(
    api_key: str | None = None,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """Download EUR/USD daily data using an environment-provided API key."""
    _ensure_dirs()

    key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")

    if not key or key == "YOUR_API_KEY":
        raise RuntimeError(
            "Set ALPHAVANTAGE_API_KEY or pass api_key; "
            "refusing to use a placeholder key."
        )

    response = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "FX_DAILY",
            "from_symbol": "EUR",
            "to_symbol": "USD",
            "outputsize": "full",
            "apikey": key,
        },
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()

    series = payload.get("Time Series FX (Daily)")

    if not series:
        detail = payload.get("Note") or payload.get("Error Message") or payload
        raise RuntimeError(
            f"Alpha Vantage response did not contain daily data: {detail}"
        )

    frame = pd.DataFrame.from_dict(series, orient="index").reset_index()
    frame.columns = ["Date", "Open", "High", "Low", "Close"]

    frame = load_ohlc_from_frame(frame)

    output = output_file or RAW_DIR / "eur_usd_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    return frame