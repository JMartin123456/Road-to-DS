"""Reusable functions for the EUR/USD direction prediction workflow.

The original project contained many copy/pasted scripts which executed at import
time.  This module keeps the existing CSV/figure outputs while making each
stage callable, testable, and safe to import.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
RANDOM_STATE = 42

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


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def load_ohlc(path: Path) -> pd.DataFrame:
    """Load and validate a chronological OHLC data set."""
    return _validate_ohlc_frame(pd.read_csv(path))


def _validate_ohlc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an OHLC frame without writing intermediate files."""
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
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


def collect_data(api_key: str | None = None, output_file: Path | None = None) -> pd.DataFrame:
    """Download EUR/USD daily data using an environment-provided API key."""
    _ensure_dirs()
    key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
    if not key or key == "YOUR_API_KEY":
        raise RuntimeError(
            "Set ALPHAVANTAGE_API_KEY or pass api_key; refusing to use a placeholder key."
        )
    response = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "FX_DAILY", "from_symbol": "EUR", "to_symbol": "USD",
            "outputsize": "full", "apikey": key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    series = payload.get("Time Series FX (Daily)")
    if not series:
        detail = payload.get("Note") or payload.get("Error Message") or payload
        raise RuntimeError(f"Alpha Vantage response did not contain daily data: {detail}")
    frame = pd.DataFrame.from_dict(series, orient="index").reset_index()
    frame.columns = ["Date", "Open", "High", "Low", "Close"]
    frame = load_ohlc_from_frame(frame)
    output = output_file or RAW_DIR / "eur_usd_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def load_ohlc_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an already materialized OHLC frame."""
    return _validate_ohlc_frame(frame)


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


def prepare_datasets(
    frame: pd.DataFrame,
    features: Iterable[str],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop unusable rows and create chronological train/validation/test splits."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave test rows.")
    feature_list = list(features)
    missing = sorted(set(feature_list + ["Target", "Date"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(missing)}")
    clean = frame.dropna(subset=feature_list + ["Target"]).copy()
    clean["Target"] = clean["Target"].astype(int)
    clean["Date"] = pd.to_datetime(clean["Date"], errors="raise")
    clean = clean.sort_values("Date").reset_index(drop=True)
    if len(clean) < 3:
        raise ValueError("At least three usable rows are required.")
    train_end = int(len(clean) * train_fraction)
    validation_end = int(len(clean) * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= len(clean):
        raise ValueError("Split fractions produce an empty data split.")
    parts = (
        clean.iloc[:train_end].copy(),
        clean.iloc[train_end:validation_end].copy(),
        clean.iloc[validation_end:].copy(),
        clean,
    )
    return parts


def dataset_paths(advanced: bool = False) -> tuple[Path, Path, Path]:
    suffix = "_advanced" if advanced else ""
    return tuple(PROCESSED_DIR / f"{name}{suffix}.csv" for name in ("train", "validation", "test"))


def load_splits(advanced: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = dataset_paths(advanced)
    frames = tuple(pd.read_csv(path) for path in paths)
    for frame, path in zip(frames, paths):
        if "Target" not in frame or frame["Target"].isna().any():
            raise ValueError(f"{path.name} contains missing targets.")
    return frames


def _check_features(frames: Iterable[pd.DataFrame], feature_list: Iterable[str]) -> None:
    required = set(feature_list) | {"Target"}
    for frame in frames:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")
        values = frame[list(feature_list)]
        if values.isna().any().any():
            raise ValueError("Model input contains missing feature values.")
        try:
            finite = np.isfinite(values.astype(float).to_numpy()).all()
        except (TypeError, ValueError) as exc:
            raise ValueError("Model features must be numeric.") from exc
        if not finite:
            raise ValueError("Model input contains infinite feature values.")


def metrics(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC_AUC": np.nan,
    }
    if y_true.nunique() > 1:
        result["ROC_AUC"] = roc_auc_score(y_true, probabilities)
    return result


def model_factories() -> Mapping[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        ),
    }


def compare_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_list: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    feature_list = list(feature_list)
    _check_features((train, validation, test), feature_list)
    X_train, y_train = train[feature_list], train["Target"]
    results = {"Validation": [], "Test": []}
    probabilities = {"Validation": {}, "Test": {}}
    fitted = {}
    for name, model in model_factories().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        for dataset_name, frame in (("Validation", validation), ("Test", test)):
            probabilities[dataset_name][name] = model.predict_proba(frame[feature_list])[:, 1]
            predictions = (probabilities[dataset_name][name] >= 0.5).astype(int)
            row = {"Model": name, **metrics(frame["Target"], predictions, probabilities[dataset_name][name])}
            results[dataset_name].append(row)
    validation_result = pd.DataFrame(results["Validation"]).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    test_result = pd.DataFrame(results["Test"]).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return validation_result, test_result, probabilities, fitted


def evaluate_thresholds(
    y_true: pd.Series, probabilities: np.ndarray, model_name: str, dataset_name: str,
    thresholds: Iterable[float] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        result = metrics(y_true, predictions, probabilities)
        rows.append({
            "Model": model_name, "Dataset": dataset_name, "Threshold": threshold,
            **{key: result[key] for key in ("Accuracy", "Precision", "Recall", "F1")},
            "Predicted_UP": int(predictions.sum()),
            "Predicted_DOWN": int(len(predictions) - predictions.sum()),
        })
    return pd.DataFrame(rows)


def tune_model(
    estimator_class: type, parameter_combinations: Iterable[dict],
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    feature_list: Iterable[str], model_name: str,
) -> tuple[pd.DataFrame, object, dict, dict]:
    feature_list = list(feature_list)
    _check_features((train, validation, test), feature_list)
    rows, best_model, best_params, best_auc = [], None, None, -np.inf
    for parameters in parameter_combinations:
        model = estimator_class(**parameters, random_state=RANDOM_STATE, **(
            {"n_jobs": -1} if estimator_class is RandomForestClassifier else {}
        ))
        model.fit(train[feature_list], train["Target"])
        probabilities = model.predict_proba(validation[feature_list])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        row = dict(parameters)
        row.update({f"Validation_{key}": value for key, value in metrics(
            validation["Target"], predictions, probabilities
        ).items()})
        rows.append(row)
        if row["Validation_ROC_AUC"] > best_auc:
            best_auc, best_model, best_params = row["Validation_ROC_AUC"], model, dict(parameters)
    result = pd.DataFrame(rows).sort_values("Validation_ROC_AUC", ascending=False).reset_index(drop=True)
    final = {}
    for name, frame in (("Validation", validation), ("Test", test)):
        probabilities = best_model.predict_proba(frame[feature_list])[:, 1]
        final[name] = metrics(frame["Target"], (probabilities >= 0.5).astype(int), probabilities)
    return result, best_model, best_params, final


def feature_selection(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    feature_list: Iterable[str], advanced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    feature_list = list(feature_list)
    _check_features((train, validation, test), feature_list)
    rf = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=RANDOM_STATE, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE)
    rf.fit(train[feature_list], train["Target"])
    gb.fit(train[feature_list], train["Target"])
    rf_perm = permutation_importance(
        rf, validation[feature_list], validation["Target"], scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = pd.DataFrame({
        "Feature": feature_list, "Importance_Mean": rf_perm.importances_mean,
        "Importance_Std": rf_perm.importances_std,
        "GB_Importance": gb.feature_importances_,
    })
    importance["Absolute_Importance"] = importance["Importance_Mean"].abs()
    importance["Average_Importance"] = (
        importance["Absolute_Importance"] + importance["GB_Importance"]
    ) / 2
    importance = importance.sort_values("Average_Importance", ascending=False).reset_index(drop=True)
    sizes = (
        [len(feature_list), 40, 30, 20, 15, 10]
        if advanced
        else [len(feature_list), 12, 10, 8, 5]
    )
    sizes = sorted({size for size in sizes if size <= len(feature_list)}, reverse=True)
    feature_sets = {"All " + str(len(feature_list)): feature_list}
    feature_sets.update({f"Top {size}": importance.head(size)["Feature"].tolist() for size in sizes if size < len(feature_list)})
    rows = []
    for set_name, selected in feature_sets.items():
        row = {"Feature_Set": set_name, "Number_of_Features": len(selected)}
        for model_name, estimator in (
            ("RF", RandomForestClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE, n_jobs=-1)),
            ("GB", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE)),
        ):
            estimator.fit(train[selected], train["Target"])
            val_prob = estimator.predict_proba(validation[selected])[:, 1]
            test_prob = estimator.predict_proba(test[selected])[:, 1]
            row[f"{model_name}_Validation_ROC_AUC"] = roc_auc_score(validation["Target"], val_prob)
            row[f"{model_name}_Test_ROC_AUC"] = roc_auc_score(test["Target"], test_prob)
        rows.append(row)
    result = pd.DataFrame(rows)
    best = {
        "RF": result.loc[result["RF_Validation_ROC_AUC"].idxmax(), "Feature_Set"],
        "GB": result.loc[result["GB_Validation_ROC_AUC"].idxmax(), "Feature_Set"],
    }
    selected = {key: feature_sets[value] for key, value in best.items()}
    return importance, result, selected


def diagnostics(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_list: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Create diagnostic tables used by the original diagnostics script."""
    feature_list = list(feature_list)
    _check_features((train, validation, test), feature_list)
    y_train, y_validation = train["Target"], validation["Target"]
    correlation = train[feature_list + ["Target"]].corr()["Target"].drop("Target")
    correlation_table = pd.DataFrame({"Feature": correlation.index, "Correlation": correlation.values})
    correlation_table["Absolute_Correlation"] = correlation_table["Correlation"].abs()
    correlation_table = correlation_table.sort_values("Absolute_Correlation", ascending=False).reset_index(drop=True)
    statistics = pd.DataFrame({
        "Feature": feature_list,
        "Mean_DOWN": [train.loc[y_train == 0, f].mean() for f in feature_list],
        "Mean_UP": [train.loc[y_train == 1, f].mean() for f in feature_list],
        "Std_DOWN": [train.loc[y_train == 0, f].std() for f in feature_list],
        "Std_UP": [train.loc[y_train == 1, f].std() for f in feature_list],
    })
    statistics["Difference"] = statistics["Mean_UP"] - statistics["Mean_DOWN"]
    statistics["Absolute_Difference"] = statistics["Difference"].abs()
    statistics = statistics.sort_values("Absolute_Difference", ascending=False).reset_index(drop=True)
    effects = statistics[["Feature", "Difference", "Std_DOWN", "Std_UP"]].copy()
    pooled = np.sqrt((effects["Std_DOWN"] ** 2 + effects["Std_UP"] ** 2) / 2)
    effects["Effect_Size"] = effects["Difference"].div(pooled.replace(0, np.nan)).fillna(0)
    effects["Absolute_Effect_Size"] = effects["Effect_Size"].abs()
    effects = effects.sort_values("Absolute_Effect_Size", ascending=False).reset_index(drop=True)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    logistic.fit(train[feature_list], y_train)
    coefficients = pd.DataFrame({"Feature": feature_list, "Coefficient": logistic[-1].coef_[0]})
    coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values("Absolute_Coefficient", ascending=False).reset_index(drop=True)
    rf = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train[feature_list], y_train)
    perm = permutation_importance(rf, validation[feature_list], y_validation, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    permutation_table = pd.DataFrame({"Feature": feature_list, "Importance_Mean": perm.importances_mean, "Importance_Std": perm.importances_std})
    permutation_table["Absolute_Importance"] = permutation_table["Importance_Mean"].abs()
    permutation_table = permutation_table.sort_values("Importance_Mean", ascending=False).reset_index(drop=True)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    gb.fit(train[feature_list], y_train)
    gb_table = pd.DataFrame({"Feature": feature_list, "Importance": gb.feature_importances_}).sort_values("Importance", ascending=False).reset_index(drop=True)
    distribution = pd.DataFrame({"Feature": feature_list})
    for name, frame in (("Train", train), ("Validation", validation), ("Test", test)):
        distribution[f"{name}_Mean"] = [frame[f].mean() for f in feature_list]
        distribution[f"{name}_Std"] = [frame[f].std() for f in feature_list]
    distribution["Validation_Mean_Diff"] = (distribution["Train_Mean"] - distribution["Validation_Mean"]).abs()
    distribution["Test_Mean_Diff"] = (distribution["Train_Mean"] - distribution["Test_Mean"]).abs()
    distribution = distribution.sort_values("Test_Mean_Diff", ascending=False).reset_index(drop=True)
    model_rows = []
    majority = y_train.value_counts().idxmax()
    for name, model in (("Logistic Regression", logistic), ("Random Forest", rf), ("Gradient Boosting", gb)):
        probability = model.predict_proba(validation[feature_list])[:, 1]
        model_rows.append({"Model": name, **{f"Validation_{key}": value for key, value in metrics(y_validation, (probability >= 0.5).astype(int), probability).items()}})
    model_rows.insert(0, {"Model": "Naive Baseline", "Validation_Accuracy": accuracy_score(y_validation, np.full(len(y_validation), majority)), "Validation_ROC_AUC": 0.5})
    summary = pd.DataFrame(model_rows)
    return {
        "diagnostics_feature_target_correlation.csv": correlation_table,
        "diagnostics_feature_statistics.csv": statistics,
        "diagnostics_effect_sizes.csv": effects,
        "diagnostics_logistic_coefficients.csv": coefficients,
        "diagnostics_permutation_importance.csv": permutation_table,
        "diagnostics_gradient_boosting_importance.csv": gb_table,
        "diagnostics_distribution_shift.csv": distribution,
        "diagnostics_model_summary.csv": summary,
    }


def save_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    _ensure_dirs()
    for filename, table in tables.items():
        table.to_csv(PROCESSED_DIR / filename, index=False)
