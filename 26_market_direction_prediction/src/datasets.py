"""Dataset preparation and validation for the EUR/USD prediction project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


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
        raise ValueError(
            "Train and validation fractions must leave test rows."
        )

    feature_list = list(features)

    missing = sorted(
        set(feature_list + ["Target", "Date"]) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing dataset columns: {', '.join(missing)}"
        )

    clean = frame.dropna(
        subset=feature_list + ["Target"]
    ).copy()

    clean["Target"] = clean["Target"].astype(int)
    clean["Date"] = pd.to_datetime(
        clean["Date"],
        errors="raise",
    )

    clean = (
        clean
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(clean) < 3:
        raise ValueError(
            "At least three usable rows are required."
        )

    train_end = int(len(clean) * train_fraction)
    validation_end = int(
        len(clean) * (train_fraction + validation_fraction)
    )

    if (
        train_end < 1
        or validation_end <= train_end
        or validation_end >= len(clean)
    ):
        raise ValueError(
            "Split fractions produce an empty data split."
        )

    train = clean.iloc[:train_end].copy()
    validation = clean.iloc[train_end:validation_end].copy()
    test = clean.iloc[validation_end:].copy()

    return train, validation, test, clean


def dataset_paths(
    advanced: bool = False,
) -> tuple[Path, Path, Path]:
    """Return paths to train, validation and test datasets."""
    suffix = "_advanced" if advanced else ""

    return tuple(
        PROCESSED_DIR / f"{name}{suffix}.csv"
        for name in ("train", "validation", "test")
    )


def load_splits(
    advanced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation and test datasets."""
    paths = dataset_paths(advanced)

    frames = tuple(
        pd.read_csv(path)
        for path in paths
    )

    for frame, path in zip(frames, paths):
        if "Target" not in frame:
            raise ValueError(
                f"{path.name} does not contain a Target column."
            )

        if frame["Target"].isna().any():
            raise ValueError(
                f"{path.name} contains missing targets."
            )

    return frames


def check_features(
    frames: Iterable[pd.DataFrame],
    feature_list: Iterable[str],
) -> None:
    """Validate that model input contains usable numeric features."""
    feature_list = list(feature_list)
    required = set(feature_list) | {"Target"}

    for frame in frames:
        missing = sorted(
            required - set(frame.columns)
        )

        if missing:
            raise ValueError(
                f"Missing features: {', '.join(missing)}"
            )

        values = frame[feature_list]

        if values.isna().any().any():
            raise ValueError(
                "Model input contains missing feature values."
            )

        try:
            finite = np.isfinite(
                values.astype(float).to_numpy()
            ).all()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Model features must be numeric."
            ) from exc

        if not finite:
            raise ValueError(
                "Model input contains infinite feature values."
            )