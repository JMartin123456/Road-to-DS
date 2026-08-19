from pathlib import Path

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

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ====================================
# Load feature data
# ====================================

input_file = (
    PROCESSED_DATA_DIR
    / "eur_usd_features.csv"
)

df = pd.read_csv(
    input_file
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)


# ====================================
# Dataset overview
# ====================================

print("\nDataset overview:")

print(
    df.info()
)

print("\nShape:")

print(
    df.shape
)

print("\nColumns:")

print(
    df.columns.tolist()
)


# ====================================
# Missing values - before cleaning
# ====================================

print("\nMissing values before cleaning:")

print(
    df.isna().sum()
)


# ====================================
# Remove rows with missing values
# ====================================

df = df.dropna().reset_index(
    drop=True
)


# ====================================
# Missing values - after cleaning
# ====================================

# print("\nMissing values after cleaning:")

# print(
#     df.isna().sum()
# )


# ====================================
# Dataset shape after cleaning
# ====================================

# print("\nShape after cleaning:")

# print(
#     df.shape
# )


# ====================================
# Target distribution
# ====================================

# print("\nTarget distribution:")

# print(
#     df["Target"]
#     .value_counts()
# )


# print("\nTarget distribution (%):")

# print(
#     df["Target"]
#     .value_counts(
#         normalize=True
#     ) * 100
# )


# ====================================
# Date range
# ====================================

# print("\nDate range after cleaning:")

# print(
#     "Start:",
#     df["Date"].min()
# )

# print(
#     "End:",
#     df["Date"].max()
# )


# ====================================
# Duplicate dates
# ====================================

# print("\nDuplicate dates:")

# print(
#     df["Date"].duplicated().sum()
# )


# ====================================
# Final dataset preview
# ====================================

# print("\nFirst rows:")

# print(
#     df.head()
# )


# print("\nLast rows:")

# print(
#     df.tail()
# )

# ====================================
# Feature / Target separation
# ====================================

feature_columns = [
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "Volatility_5",
    "Volatility_20",
    "Volatility_50",
    "Volatility_Ratio_5_20",
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "Price_vs_SMA_20",
    "High_Low_Range_Pct",
    "Open_Close_Range_Pct",
    "Body_Size_Pct",
    "Upper_Wick_Pct",
    "Lower_Wick_Pct"
]


X = df[
    feature_columns
]

y = df[
    "Target"
]


# ====================================
# Check X and y
# ====================================

print("\nFeatures (X):")

print(
    X.head()
)

print("\nX shape:")

print(
    X.shape
)

print("\nTarget (y):")

print(
    y.head()
)

print("\ny shape:")

print(
    y.shape
)


print("\nFeature columns:")

print(
    X.columns.tolist()
)

print("\nX missing values:")
print(X.isna().sum().sum())

print("\ny missing values:")
print(y.isna().sum())

# ====================================
# Train / Validation / Test split
# ====================================

n = len(df)

train_end = int(
    n * 0.70
)

validation_end = int(
    n * 0.85
)


train_df = df.iloc[
    :train_end
].copy()

validation_df = df.iloc[
    train_end:validation_end
].copy()

test_df = df.iloc[
    validation_end:
].copy()

# ====================================
# X / y split
# ====================================

X_train = train_df[
    feature_columns
]

y_train = train_df[
    "Target"
]


X_validation = validation_df[
    feature_columns
]

y_validation = validation_df[
    "Target"
]


X_test = test_df[
    feature_columns
]

y_test = test_df[
    "Target"
]

# ====================================
# Split overview
# ====================================

print("\nDataset split:")

print(
    "Train:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_validation.shape,
    y_validation.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)

print("\nDate ranges:")

print(
    "Train:",
    train_df["Date"].min(),
    "->",
    train_df["Date"].max()
)

print(
    "Validation:",
    validation_df["Date"].min(),
    "->",
    validation_df["Date"].max()
)

print(
    "Test:",
    test_df["Date"].min(),
    "->",
    test_df["Date"].max()
)

print("\nTarget distribution:")

print("\nTrain:")
print(
    y_train.value_counts(
        normalize=True
    ) * 100
)

print("\nValidation:")
print(
    y_validation.value_counts(
        normalize=True
    ) * 100
)

print("\nTest:")
print(
    y_test.value_counts(
        normalize=True
    ) * 100
)

print("\nDate overlap checks:")

print(
    "Train -> Validation:",
    train_df["Date"].max()
    >= validation_df["Date"].min()
)

print(
    "Validation -> Test:",
    validation_df["Date"].max()
    >= test_df["Date"].min()
)

# ====================================
# Save datasets
# ====================================

train_file = (
    PROCESSED_DATA_DIR
    / "train.csv"
)

validation_file = (
    PROCESSED_DATA_DIR
    / "validation.csv"
)

test_file = (
    PROCESSED_DATA_DIR
    / "test.csv"
)


train_df.to_csv(
    train_file,
    index=False
)

validation_df.to_csv(
    validation_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)

print("\nDatasets saved:")

print(
    train_file
)

print(
    validation_file
)

print(
    test_file
)

# ====================================
# Save cleaned dataset
# ====================================

output_file = (
    PROCESSED_DATA_DIR
    / "eur_usd_model_data.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ====================================
# Summary
# ====================================

print("\nDataset preparation completed.")

print(
    f"Saved to: {output_file}"
)

print(
    "Final shape:",
    df.shape
)