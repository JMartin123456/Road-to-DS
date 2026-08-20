from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.inspection import permutation_importance

from sklearn.metrics import roc_auc_score


# ====================================
# CONFIG
# ====================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


RANDOM_STATE = 42


# ====================================
# LOAD DATASETS
# ====================================

print("\n" + "=" * 70)
print("LOADING ADVANCED DATASETS")
print("=" * 70)


train = pd.read_csv(
    PROCESSED_DATA_DIR
    / "train_advanced.csv"
)

validation = pd.read_csv(
    PROCESSED_DATA_DIR
    / "validation_advanced.csv"
)

test = pd.read_csv(
    PROCESSED_DATA_DIR
    / "test_advanced.csv"
)


print("\nTrain shape:")
print(train.shape)

print("\nValidation shape:")
print(validation.shape)

print("\nTest shape:")
print(test.shape)


# ====================================
# FEATURE COLUMNS
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

    "Upper_Wick",
    "Lower_Wick",
    "Upper_Wick_Pct",
    "Lower_Wick_Pct",

    "Momentum_3d",
    "Momentum_10d",
    "Momentum_30d",
    "Momentum_Acceleration",

    "Return_5d_change",

    "Price_vs_SMA_5",
    "Price_vs_SMA_50",

    "SMA_5_vs_SMA_20",
    "SMA_20_vs_SMA_50",

    "SMA_5_slope",
    "SMA_20_slope",
    "SMA_50_slope",

    "Volatility_5_vs_20",
    "Volatility_20_vs_50",
    "Volatility_5_change",
    "Volatility_20_change",

    "Range_5d",
    "Range_20d",
    "Range_50d",

    "Price_Position_20",
    "Price_Position_50",

    "Distance_From_High_20",
    "Distance_From_Low_20",

    "Body_to_Range",
    "Upper_Wick_to_Range",
    "Lower_Wick_to_Range",

    "Candle_Direction",
    "Previous_Candle_Direction",
    "Previous_Return",

    "Return_Mean_5",
    "Return_Mean_20",

    "Return_Std_5",
    "Return_Std_20",

    "Positive_Return_Ratio_10",
    "Positive_Return_Ratio_20",

    "Momentum_5_Vol_Adjusted",
    "Momentum_20_Vol_Adjusted",
]


print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(
    f"\nNumber of features: "
    f"{len(feature_columns)}"
)

for index, feature in enumerate(
    feature_columns,
    start=1,
):
    print(
        f"{index:02d}. {feature}"
    )


# ====================================
# CHECK FEATURES
# ====================================

missing_train = [
    feature
    for feature in feature_columns
    if feature not in train.columns
]

missing_validation = [
    feature
    for feature in feature_columns
    if feature not in validation.columns
]

missing_test = [
    feature
    for feature in feature_columns
    if feature not in test.columns
]


if missing_train:
    raise ValueError(
        "Missing features in train dataset:\n"
        + "\n".join(missing_train)
    )


if missing_validation:
    raise ValueError(
        "Missing features in validation dataset:\n"
        + "\n".join(missing_validation)
    )


if missing_test:
    raise ValueError(
        "Missing features in test dataset:\n"
        + "\n".join(missing_test)
    )


# ====================================
# X / y
# ====================================

X_train = train[
    feature_columns
]

y_train = train[
    "Target"
]

X_validation = validation[
    feature_columns
]

y_validation = validation[
    "Target"
]

X_test = test[
    feature_columns
]

y_test = test[
    "Target"
]


# ====================================
# TARGET CHECK
# ====================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)


for name, target in [
    ("Train", y_train),
    ("Validation", y_validation),
    ("Test", y_test),
]:

    print(f"\n{name}")

    print(
        target
        .value_counts()
        .sort_index()
        .to_string()
    )


# ====================================
# RANDOM FOREST
# PERMUTATION IMPORTANCE
# ====================================

print("\n" + "=" * 70)
print("RANDOM FOREST PERMUTATION IMPORTANCE")
print("=" * 70)


rf_importance_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


rf_importance_model.fit(
    X_train,
    y_train,
)


rf_validation_probabilities = (
    rf_importance_model
    .predict_proba(
        X_validation
    )[:, 1]
)


rf_validation_auc = roc_auc_score(
    y_validation,
    rf_validation_probabilities,
)


print(
    f"\nRandom Forest Validation ROC-AUC: "
    f"{rf_validation_auc:.4f}"
)


rf_permutation = permutation_importance(
    rf_importance_model,
    X_validation,
    y_validation,
    scoring="roc_auc",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


rf_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance_Mean":
        rf_permutation.importances_mean,
    "Importance_Std":
        rf_permutation.importances_std,
})


rf_importance[
    "Absolute_Importance"
] = rf_importance[
    "Importance_Mean"
].abs()


rf_importance = (
    rf_importance
    .sort_values(
        "Importance_Mean",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    "\nRandom Forest permutation importance:"
)

print(
    rf_importance.to_string(
        index=False
    )
)


# ====================================
# GRADIENT BOOSTING
# ====================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING IMPORTANCE")
print("=" * 70)


gb_importance_model = (
    GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )
)


gb_importance_model.fit(
    X_train,
    y_train,
)


gb_validation_probabilities = (
    gb_importance_model
    .predict_proba(
        X_validation
    )[:, 1]
)


gb_validation_auc = roc_auc_score(
    y_validation,
    gb_validation_probabilities,
)


print(
    f"\nGradient Boosting Validation ROC-AUC: "
    f"{gb_validation_auc:.4f}"
)


gb_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance":
        gb_importance_model.feature_importances_,
})


gb_importance = (
    gb_importance
    .sort_values(
        "Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    "\nGradient Boosting feature importance:"
)

print(
    gb_importance.to_string(
        index=False
    )
)


# ====================================
# COMBINE IMPORTANCE
# ====================================

combined = pd.merge(
    rf_importance,
    gb_importance,
    on="Feature",
    how="outer",
)


combined = combined.rename(
    columns={
        "Importance":
            "GB_Importance",
    }
)


combined["RF_Rank"] = (
    combined[
        "Importance_Mean"
    ]
    .rank(
        ascending=False,
        method="min",
    )
)


combined["GB_Rank"] = (
    combined[
        "GB_Importance"
    ]
    .rank(
        ascending=False,
        method="min",
    )
)


combined["Average_Rank"] = (
    combined["RF_Rank"]
    +
    combined["GB_Rank"]
) / 2


combined["Average_Importance"] = (
    combined["Absolute_Importance"]
    +
    combined["GB_Importance"]
) / 2


combined = (
    combined
    .sort_values(
        [
            "Average_Rank",
            "Average_Importance",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)


# ====================================
# FEATURE RANKING
# ====================================

print("\n" + "=" * 70)
print("COMBINED FEATURE RANKING")
print("=" * 70)


print(
    combined.to_string(
        index=False
    )
)


# ====================================
# SAVE FEATURE IMPORTANCE
# ====================================

importance_output = (
    PROCESSED_DATA_DIR
    / "advanced_feature_importance.csv"
)


combined.to_csv(
    importance_output,
    index=False,
)


print(
    "\nFeature importance saved to:"
)

print(
    importance_output
)


# ====================================
# FEATURE SETS
# ====================================

feature_sets = {

    "All 55":
        feature_columns,

    "Top 40":
        combined[
            "Feature"
        ].head(40).tolist(),

    "Top 30":
        combined[
            "Feature"
        ].head(30).tolist(),

    "Top 20":
        combined[
            "Feature"
        ].head(20).tolist(),

    "Top 15":
        combined[
            "Feature"
        ].head(15).tolist(),

    "Top 10":
        combined[
            "Feature"
        ].head(10).tolist(),
}


# ====================================
# FEATURE SELECTION TEST
# ====================================

print("\n" + "=" * 70)
print("TESTING FEATURE SETS")
print("=" * 70)


results = []


for set_name, selected_features in feature_sets.items():

    print("\n")
    print("=" * 70)

    print(
        f"TESTING: {set_name}"
    )

    print("=" * 70)


    X_train_selected = (
        X_train[
            selected_features
        ]
    )

    X_validation_selected = (
        X_validation[
            selected_features
        ]
    )


    # ==================================
    # RANDOM FOREST
    # ==================================

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


    rf.fit(
        X_train_selected,
        y_train,
    )


    rf_validation_probabilities = (
        rf.predict_proba(
            X_validation_selected
        )[:, 1]
    )


    rf_validation_auc = (
        roc_auc_score(
            y_validation,
            rf_validation_probabilities,
        )
    )


    print(
        f"Random Forest Validation ROC-AUC: "
        f"{rf_validation_auc:.4f}"
    )


    # ==================================
    # GRADIENT BOOSTING
    # ==================================

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )


    gb.fit(
        X_train_selected,
        y_train,
    )


    gb_validation_probabilities = (
        gb.predict_proba(
            X_validation_selected
        )[:, 1]
    )


    gb_validation_auc = (
        roc_auc_score(
            y_validation,
            gb_validation_probabilities,
        )
    )


    print(
        f"Gradient Boosting Validation ROC-AUC: "
        f"{gb_validation_auc:.4f}"
    )


    # ==================================
    # SAVE RESULT
    # ==================================

    results.append({

        "Feature_Set":
            set_name,

        "Number_of_Features":
            len(selected_features),

        "RF_Validation_ROC_AUC":
            rf_validation_auc,

        "GB_Validation_ROC_AUC":
            gb_validation_auc,
    })


# ====================================
# RESULTS DATAFRAME
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
# BEST RF
# ====================================

best_rf = (
    results_df
    .sort_values(
        "RF_Validation_ROC_AUC",
        ascending=False,
    )
    .iloc[0]
)


print("\n" + "=" * 70)
print("BEST RANDOM FOREST FEATURE SET")
print("=" * 70)


print(
    best_rf.to_string()
)


# ====================================
# BEST GB
# ====================================

best_gb = (
    results_df
    .sort_values(
        "GB_Validation_ROC_AUC",
        ascending=False,
    )
    .iloc[0]
)


print("\n" + "=" * 70)
print("BEST GRADIENT BOOSTING FEATURE SET")
print("=" * 70)


print(
    best_gb.to_string()
)


# ====================================
# FINAL TEST
# ====================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


best_rf_feature_set = (
    best_rf[
        "Feature_Set"
    ]
)


best_gb_feature_set = (
    best_gb[
        "Feature_Set"
    ]
)


rf_features = feature_sets[
    best_rf_feature_set
]

gb_features = feature_sets[
    best_gb_feature_set
]


# ====================================
# BEST RF ON TEST
# ====================================

rf_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


rf_final.fit(
    X_train[rf_features],
    y_train,
)


rf_test_probabilities = (
    rf_final
    .predict_proba(
        X_test[rf_features]
    )[:, 1]
)


rf_test_auc = roc_auc_score(
    y_test,
    rf_test_probabilities,
)


print(
    f"\nBest RF feature set: "
    f"{best_rf_feature_set}"
)

print(
    f"RF Test ROC-AUC: "
    f"{rf_test_auc:.4f}"
)


# ====================================
# BEST GB ON TEST
# ====================================

gb_final = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
)


gb_final.fit(
    X_train[gb_features],
    y_train,
)


gb_test_probabilities = (
    gb_final
    .predict_proba(
        X_test[gb_features]
    )[:, 1]
)


gb_test_auc = roc_auc_score(
    y_test,
    gb_test_probabilities,
)


print(
    f"\nBest GB feature set: "
    f"{best_gb_feature_set}"
)

print(
    f"GB Test ROC-AUC: "
    f"{gb_test_auc:.4f}"
)


# ====================================
# ADD TEST RESULTS
# ====================================

results_df[
    "RF_Test_ROC_AUC"
] = None

results_df[
    "GB_Test_ROC_AUC"
] = None


results_df.loc[
    results_df[
        "Feature_Set"
    ] == best_rf_feature_set,
    "RF_Test_ROC_AUC",
] = rf_test_auc


results_df.loc[
    results_df[
        "Feature_Set"
    ] == best_gb_feature_set,
    "GB_Test_ROC_AUC",
] = gb_test_auc


# ====================================
# SAVE RESULTS
# ====================================

results_output = (
    PROCESSED_DATA_DIR
    / "advanced_feature_selection_results.csv"
)


results_df.to_csv(
    results_output,
    index=False,
)


print(
    "\nResults saved to:"
)

print(
    results_output
)


# ====================================
# SAVE SELECTED FEATURES
# ====================================

selected_features_output = (
    PROCESSED_DATA_DIR
    / "advanced_selected_features.csv"
)


selected_features_df = pd.DataFrame({

    "RF_Best_Feature_Set":
        pd.Series(
            rf_features
        ),

    "GB_Best_Feature_Set":
        pd.Series(
            gb_features
        ),
})


selected_features_df.to_csv(
    selected_features_output,
    index=False,
)


print(
    "\nSelected features saved to:"
)

print(
    selected_features_output
)


# ====================================
# FINAL SUMMARY
# ====================================

print("\n" + "=" * 70)
print("ADVANCED FEATURE SELECTION SUMMARY")
print("=" * 70)


print(
    f"\nNumber of original features: "
    f"{len(feature_columns)}"
)


print(
    f"\nBest RF feature set: "
    f"{best_rf_feature_set}"
)


print(
    f"Best RF validation ROC-AUC: "
    f"{best_rf['RF_Validation_ROC_AUC']:.4f}"
)


print(
    f"Best RF test ROC-AUC: "
    f"{rf_test_auc:.4f}"
)


print(
    f"\nBest GB feature set: "
    f"{best_gb_feature_set}"
)


print(
    f"Best GB validation ROC-AUC: "
    f"{best_gb['GB_Validation_ROC_AUC']:.4f}"
)


print(
    f"Best GB test ROC-AUC: "
    f"{gb_test_auc:.4f}"
)


print("\n" + "=" * 70)
print("ADVANCED FEATURE SELECTION COMPLETED")
print("=" * 70)