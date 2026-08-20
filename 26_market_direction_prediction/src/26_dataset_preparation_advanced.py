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

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "eur_usd_features_advanced.csv"
)

TRAIN_FILE = (
    PROCESSED_DATA_DIR
    / "train_advanced.csv"
)

VALIDATION_FILE = (
    PROCESSED_DATA_DIR
    / "validation_advanced.csv"
)

TEST_FILE = (
    PROCESSED_DATA_DIR
    / "test_advanced.csv"
)

FINAL_FILE = (
    PROCESSED_DATA_DIR
    / "eur_usd_model_data_advanced.csv"
)


# ====================================
# Load dataset
# ====================================

print("\n========================================")
print("Loading advanced feature dataset")
print("========================================")

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
# Feature columns
# ====================================

feature_columns = [

    # --------------------------------
    # Original features
    # --------------------------------

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
    "Lower_Wick_Pct",

    # --------------------------------
    # Advanced momentum
    # --------------------------------

    "Momentum_3d",
    "Momentum_10d",
    "Momentum_30d",

    "Momentum_Acceleration",
    "Return_5d_change",

    # --------------------------------
    # Advanced trend
    # --------------------------------

    "Price_vs_SMA_5",
    "Price_vs_SMA_50",

    "SMA_5_vs_SMA_20",
    "SMA_20_vs_SMA_50",

    "SMA_5_slope",
    "SMA_20_slope",
    "SMA_50_slope",

    # --------------------------------
    # Advanced volatility
    # --------------------------------

    "Volatility_5_vs_20",
    "Volatility_20_vs_50",

    "Volatility_5_change",
    "Volatility_20_change",

    # --------------------------------
    # Price range
    # --------------------------------

    "Range_5d",
    "Range_20d",
    "Range_50d",

    # --------------------------------
    # Price position
    # --------------------------------

    "Price_Position_20",
    "Price_Position_50",

    "Distance_From_High_20",
    "Distance_From_Low_20",

    # --------------------------------
    # Candle structure
    # --------------------------------

    "Body_to_Range",
    "Upper_Wick_to_Range",
    "Lower_Wick_to_Range",

    "Candle_Direction",
    "Previous_Candle_Direction",
    "Previous_Return",

    # --------------------------------
    # Rolling statistics
    # --------------------------------

    "Return_Mean_5",
    "Return_Mean_20",

    "Return_Std_5",
    "Return_Std_20",

    "Positive_Return_Ratio_10",
    "Positive_Return_Ratio_20",

    # --------------------------------
    # Volatility-adjusted momentum
    # --------------------------------

    "Momentum_5_Vol_Adjusted",
    "Momentum_20_Vol_Adjusted",
]


# ====================================
# Check feature columns
# ====================================

print("\nNumber of feature columns:")
print(len(feature_columns))

missing_feature_columns = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_feature_columns:

    print("\nERROR: Missing feature columns:")

    print(
        missing_feature_columns
    )

    raise ValueError(
        "Some feature columns are missing from the dataset."
    )


print("\nAll feature columns are present.")


# ====================================
# X and y before cleaning
# ====================================

X = df[
    feature_columns
]

y = df[
    "Target"
]


# ====================================
# Missing values
# ====================================

print("\n========================================")
print("Missing values before cleaning")
print("========================================")

missing_values = (
    X
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    missing_values[
        missing_values > 0
    ]
)


print("\nTotal rows before cleaning:")
print(len(df))


# ====================================
# Remove rows with missing features
# ====================================

valid_rows = (
    X
    .notna()
    .all(axis=1)
)

df = (
    df.loc[valid_rows]
    .reset_index(drop=True)
)


print("\nTotal rows after cleaning:")
print(len(df))

print("\nRows removed:")
print(
    5000 - len(df)
)


# ====================================
# Verify no missing values
# ====================================

X = df[
    feature_columns
]

y = df[
    "Target"
]


print("\n========================================")
print("Missing values after cleaning")
print("========================================")

print(
    X.isna().sum().sum()
)

print(
    "\nX missing values:",
    X.isna().sum().sum()
)

print(
    "y missing values:",
    y.isna().sum()
)


# ====================================
# Chronological split
# ====================================

print("\n========================================")
print("Dataset split")
print("========================================")

total_rows = len(df)

train_end = int(
    total_rows * 0.70
)

validation_end = int(
    total_rows * 0.85
)


train = (
    df
    .iloc[:train_end]
    .copy()
)

validation = (
    df
    .iloc[train_end:validation_end]
    .copy()
)

test = (
    df
    .iloc[validation_end:]
    .copy()
)


# ====================================
# Split shapes
# ====================================

print("\nTrain:")
print(train.shape)

print("\nValidation:")
print(validation.shape)

print("\nTest:")
print(test.shape)


# ====================================
# Date ranges
# ====================================

print("\n========================================")
print("Date ranges")
print("========================================")

print(
    "Train:",
    train["Date"].min(),
    "->",
    train["Date"].max()
)

print(
    "Validation:",
    validation["Date"].min(),
    "->",
    validation["Date"].max()
)

print(
    "Test:",
    test["Date"].min(),
    "->",
    test["Date"].max()
)


# ====================================
# Check date overlap
# ====================================

print("\n========================================")
print("Date overlap checks")
print("========================================")

train_validation_overlap = (
    train["Date"].max()
    >=
    validation["Date"].min()
)

validation_test_overlap = (
    validation["Date"].max()
    >=
    test["Date"].min()
)

print(
    "Train -> Validation:",
    train_validation_overlap
)

print(
    "Validation -> Test:",
    validation_test_overlap
)


# ====================================
# Target distribution
# ====================================

print("\n========================================")
print("Target distribution")
print("========================================")


def print_target_distribution(
    dataset,
    dataset_name
):

    print(f"\n{dataset_name}:")

    distribution = (
        dataset["Target"]
        .value_counts(
            normalize=True
        )
        * 100
    )

    print(
        distribution
    )


print_target_distribution(
    train,
    "Train"
)

print_target_distribution(
    validation,
    "Validation"
)

print_target_distribution(
    test,
    "Test"
)


# ====================================
# Feature check
# ====================================

print("\n========================================")
print("Feature check")
print("========================================")

print(
    "Number of features:",
    len(feature_columns)
)

print(
    "\nFeature columns:"
)

for index, column in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{index:02d}. {column}"
    )


# ====================================
# Save datasets
# ====================================

train.to_csv(
    TRAIN_FILE,
    index=False
)

validation.to_csv(
    VALIDATION_FILE,
    index=False
)

test.to_csv(
    TEST_FILE,
    index=False
)

df.to_csv(
    FINAL_FILE,
    index=False
)


# ====================================
# Final verification
# ====================================

print("\n========================================")
print("Datasets saved")
print("========================================")

print(
    "\nTrain:"
)

print(
    TRAIN_FILE
)

print(
    "\nValidation:"
)

print(
    VALIDATION_FILE
)

print(
    "\nTest:"
)

print(
    TEST_FILE
)

print(
    "\nFull advanced dataset:"
)

print(
    FINAL_FILE
)


# ====================================
# Final summary
# ====================================

print("\n========================================")
print("ADVANCED DATASET PREPARATION COMPLETED")
print("========================================")

print(
    "\nFinal shape:",
    df.shape
)

print(
    "Number of features:",
    len(feature_columns)
)

print(
    "Train rows:",
    len(train)
)

print(
    "Validation rows:",
    len(validation)
)

print(
    "Test rows:",
    len(test)
)

print(
    "\nOutput files:"
)

print(
    "- train_advanced.csv"
)

print(
    "- validation_advanced.csv"
)

print(
    "- test_advanced.csv"
)

print(
    "- eur_usd_model_data_advanced.csv"
)