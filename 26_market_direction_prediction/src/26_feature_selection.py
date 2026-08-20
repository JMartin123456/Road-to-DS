from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)


# ====================================
# Config
# ====================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# ====================================
# Load datasets
# ====================================

train = pd.read_csv(
    PROCESSED_DATA_DIR / "train.csv"
)

validation = pd.read_csv(
    PROCESSED_DATA_DIR / "validation.csv"
)

test = pd.read_csv(
    PROCESSED_DATA_DIR / "test.csv"
)


# ====================================
# Feature columns
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
    "Lower_Wick_Pct",
]


# ====================================
# X and y
# ====================================

X_train = train[feature_columns]
y_train = train["Target"]

X_validation = validation[feature_columns]
y_validation = validation["Target"]

X_test = test[feature_columns]
y_test = test["Target"]


# ====================================
# Random Forest feature importance
# ====================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=3,
    random_state=42,
)

rf.fit(
    X_train,
    y_train
)

rf_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rf.feature_importances_,
})

rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

print(rf_importance)


# ====================================
# Gradient Boosting feature importance
# ====================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING FEATURE IMPORTANCE")
print("=" * 70)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=2,
    random_state=42,
)

gb.fit(
    X_train,
    y_train
)

gb_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": gb.feature_importances_,
})

gb_importance = gb_importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

print(gb_importance)


# ====================================
# Average feature importance
# ====================================

importance = pd.merge(
    rf_importance,
    gb_importance,
    on="Feature",
    suffixes=("_RF", "_GB")
)

importance["Average_Importance"] = (
    importance["Importance_RF"]
    + importance["Importance_GB"]
) / 2

importance = importance.sort_values(
    "Average_Importance",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("COMBINED FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance[
        [
            "Feature",
            "Importance_RF",
            "Importance_GB",
            "Average_Importance",
        ]
    ]
)


# ====================================
# Feature groups
# ====================================

ranked_features = importance["Feature"].tolist()

feature_sets = {
    "All 16": ranked_features,
    "Top 12": ranked_features[:12],
    "Top 10": ranked_features[:10],
    "Top 8": ranked_features[:8],
    "Top 5": ranked_features[:5],
}


# ====================================
# Test feature combinations
# ====================================

results = []


for set_name, features in feature_sets.items():

    print("\n" + "=" * 70)
    print(f"TESTING: {set_name}")
    print("=" * 70)

    X_train_subset = X_train[features]
    X_validation_subset = X_validation[features]
    X_test_subset = X_test[features]

    # --------------------------------
    # Random Forest
    # --------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=42,
    )

    rf_model.fit(
        X_train_subset,
        y_train
    )

    rf_validation_probabilities = rf_model.predict_proba(
        X_validation_subset
    )[:, 1]

    rf_test_probabilities = rf_model.predict_proba(
        X_test_subset
    )[:, 1]

    rf_validation_auc = roc_auc_score(
        y_validation,
        rf_validation_probabilities
    )

    rf_test_auc = roc_auc_score(
        y_test,
        rf_test_probabilities
    )

    # --------------------------------
    # Gradient Boosting
    # --------------------------------

    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )

    gb_model.fit(
        X_train_subset,
        y_train
    )

    gb_validation_probabilities = gb_model.predict_proba(
        X_validation_subset
    )[:, 1]

    gb_test_probabilities = gb_model.predict_proba(
        X_test_subset
    )[:, 1]

    gb_validation_auc = roc_auc_score(
        y_validation,
        gb_validation_probabilities
    )

    gb_test_auc = roc_auc_score(
        y_test,
        gb_test_probabilities
    )

    # --------------------------------
    # Save results
    # --------------------------------

    results.append({
        "Feature_Set": set_name,
        "Number_of_Features": len(features),
        "RF_Validation_ROC_AUC": rf_validation_auc,
        "RF_Test_ROC_AUC": rf_test_auc,
        "GB_Validation_ROC_AUC": gb_validation_auc,
        "GB_Test_ROC_AUC": gb_test_auc,
    })

    print(
        f"Random Forest Validation ROC-AUC: "
        f"{rf_validation_auc:.4f}"
    )

    print(
        f"Random Forest Test ROC-AUC: "
        f"{rf_test_auc:.4f}"
    )

    print(
        f"Gradient Boosting Validation ROC-AUC: "
        f"{gb_validation_auc:.4f}"
    )

    print(
        f"Gradient Boosting Test ROC-AUC: "
        f"{gb_test_auc:.4f}"
    )


# ====================================
# Results
# ====================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ====================================
# Best configurations
# ====================================

best_rf = results_df.loc[
    results_df["RF_Validation_ROC_AUC"].idxmax()
]

best_gb = results_df.loc[
    results_df["GB_Validation_ROC_AUC"].idxmax()
]


print("\n" + "=" * 70)
print("BEST RANDOM FOREST FEATURE SET")
print("=" * 70)

print(
    best_rf
)


print("\n" + "=" * 70)
print("BEST GRADIENT BOOSTING FEATURE SET")
print("=" * 70)

print(
    best_gb
)


# ====================================
# Save results
# ====================================

importance_file = (
    PROCESSED_DATA_DIR
    / "feature_importance.csv"
)

results_file = (
    PROCESSED_DATA_DIR
    / "feature_selection_results.csv"
)

importance.to_csv(
    importance_file,
    index=False
)

results_df.to_csv(
    results_file,
    index=False
)


# ====================================
# Summary
# ====================================

print("\n" + "=" * 70)
print("FEATURE SELECTION COMPLETED")
print("=" * 70)

print(
    f"Feature importance saved to: {importance_file}"
)

print(
    f"Feature selection results saved to: {results_file}"
)