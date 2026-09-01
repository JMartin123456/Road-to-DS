"""Analysis and diagnostic functions for the EUR/USD prediction project."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets import check_features
from evaluation import metrics


RANDOM_STATE = 42


def feature_selection(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_list: Iterable[str],
    advanced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Rank features and evaluate different feature-set sizes."""

    feature_list = list(feature_list)
    check_features((train, validation, test), feature_list)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )

    rf.fit(train[feature_list], train["Target"])
    gb.fit(train[feature_list], train["Target"])

    rf_perm = permutation_importance(
        rf,
        validation[feature_list],
        validation["Target"],
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "Feature": feature_list,
        "Importance_Mean": rf_perm.importances_mean,
        "Importance_Std": rf_perm.importances_std,
        "GB_Importance": gb.feature_importances_,
    })

    importance["Absolute_Importance"] = importance["Importance_Mean"].abs()

    importance["Average_Importance"] = (
        importance["Absolute_Importance"]
        + importance["GB_Importance"]
    ) / 2

    importance = (
        importance
        .sort_values("Average_Importance", ascending=False)
        .reset_index(drop=True)
    )

    if advanced:
        sizes = [len(feature_list), 40, 30, 20, 15, 10]
    else:
        sizes = [len(feature_list), 12, 10, 8, 5]

    sizes = sorted(
        {size for size in sizes if size <= len(feature_list)},
        reverse=True,
    )

    feature_sets = {
        "All " + str(len(feature_list)): feature_list
    }

    feature_sets.update({
        f"Top {size}": importance.head(size)["Feature"].tolist()
        for size in sizes
        if size < len(feature_list)
    })

    rows = []

    for set_name, selected in feature_sets.items():

        row = {
            "Feature_Set": set_name,
            "Number_of_Features": len(selected),
        }

        for model_name, estimator in (
            (
                "RF",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "GB",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ):

            estimator.fit(train[selected], train["Target"])

            val_prob = estimator.predict_proba(
                validation[selected]
            )[:, 1]

            test_prob = estimator.predict_proba(
                test[selected]
            )[:, 1]

            row[f"{model_name}_Validation_ROC_AUC"] = (
                pd.Series(validation["Target"])
                .pipe(
                    lambda y: __import__(
                        "sklearn.metrics",
                        fromlist=["roc_auc_score"],
                    ).roc_auc_score(y, val_prob)
                )
            )

            row[f"{model_name}_Test_ROC_AUC"] = (
                pd.Series(test["Target"])
                .pipe(
                    lambda y: __import__(
                        "sklearn.metrics",
                        fromlist=["roc_auc_score"],
                    ).roc_auc_score(y, test_prob)
                )
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    best = {
        "RF": result.loc[
            result["RF_Validation_ROC_AUC"].idxmax(),
            "Feature_Set",
        ],
        "GB": result.loc[
            result["GB_Validation_ROC_AUC"].idxmax(),
            "Feature_Set",
        ],
    }

    selected = {
        key: feature_sets[value]
        for key, value in best.items()
    }

    return importance, result, selected


def diagnostics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_list: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Create diagnostic tables for the model and feature set."""

    feature_list = list(feature_list)

    check_features(
        (train, validation, test),
        feature_list,
    )

    y_train = train["Target"]
    y_validation = validation["Target"]

    # ---------------------------------------------------------
    # Feature / target correlation
    # ---------------------------------------------------------

    correlation = (
        train[feature_list + ["Target"]]
        .corr()["Target"]
        .drop("Target")
    )

    correlation_table = pd.DataFrame({
        "Feature": correlation.index,
        "Correlation": correlation.values,
    })

    correlation_table["Absolute_Correlation"] = (
        correlation_table["Correlation"].abs()
    )

    correlation_table = (
        correlation_table
        .sort_values("Absolute_Correlation", ascending=False)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Feature statistics by target class
    # ---------------------------------------------------------

    statistics = pd.DataFrame({
        "Feature": feature_list,

        "Mean_DOWN": [
            train.loc[y_train == 0, feature].mean()
            for feature in feature_list
        ],

        "Mean_UP": [
            train.loc[y_train == 1, feature].mean()
            for feature in feature_list
        ],

        "Std_DOWN": [
            train.loc[y_train == 0, feature].std()
            for feature in feature_list
        ],

        "Std_UP": [
            train.loc[y_train == 1, feature].std()
            for feature in feature_list
        ],
    })

    statistics["Difference"] = (
        statistics["Mean_UP"]
        - statistics["Mean_DOWN"]
    )

    statistics["Absolute_Difference"] = (
        statistics["Difference"].abs()
    )

    statistics = (
        statistics
        .sort_values("Absolute_Difference", ascending=False)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Effect sizes
    # ---------------------------------------------------------

    effects = statistics[
        ["Feature", "Difference", "Std_DOWN", "Std_UP"]
    ].copy()

    pooled = np.sqrt(
        (
            effects["Std_DOWN"] ** 2
            + effects["Std_UP"] ** 2
        ) / 2
    )

    effects["Effect_Size"] = (
        effects["Difference"]
        .div(pooled.replace(0, np.nan))
        .fillna(0)
    )

    effects["Absolute_Effect_Size"] = (
        effects["Effect_Size"].abs()
    )

    effects = (
        effects
        .sort_values("Absolute_Effect_Size", ascending=False)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Logistic Regression coefficients
    # ---------------------------------------------------------

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )

    logistic.fit(
        train[feature_list],
        y_train,
    )

    coefficients = pd.DataFrame({
        "Feature": feature_list,
        "Coefficient": logistic[-1].coef_[0],
    })

    coefficients["Absolute_Coefficient"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = (
        coefficients
        .sort_values(
            "Absolute_Coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Random Forest permutation importance
    # ---------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf.fit(
        train[feature_list],
        y_train,
    )

    perm = permutation_importance(
        rf,
        validation[feature_list],
        y_validation,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    permutation_table = pd.DataFrame({
        "Feature": feature_list,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std,
    })

    permutation_table["Absolute_Importance"] = (
        permutation_table["Importance_Mean"].abs()
    )

    permutation_table = (
        permutation_table
        .sort_values(
            "Importance_Mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Gradient Boosting importance
    # ---------------------------------------------------------

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    gb.fit(
        train[feature_list],
        y_train,
    )

    gb_table = pd.DataFrame({
        "Feature": feature_list,
        "Importance": gb.feature_importances_,
    })

    gb_table = (
        gb_table
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Distribution shift
    # ---------------------------------------------------------

    distribution = pd.DataFrame({
        "Feature": feature_list
    })

    for name, frame in (
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ):
        distribution[f"{name}_Mean"] = [
            frame[feature].mean()
            for feature in feature_list
        ]

        distribution[f"{name}_Std"] = [
            frame[feature].std()
            for feature in feature_list
        ]

    distribution["Validation_Mean_Diff"] = (
        distribution["Train_Mean"]
        - distribution["Validation_Mean"]
    ).abs()

    distribution["Test_Mean_Diff"] = (
        distribution["Train_Mean"]
        - distribution["Test_Mean"]
    ).abs()

    distribution = (
        distribution
        .sort_values(
            "Test_Mean_Diff",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Model diagnostic summary
    # ---------------------------------------------------------

    model_rows = []

    majority = y_train.value_counts().idxmax()

    for name, model in (
        ("Logistic Regression", logistic),
        ("Random Forest", rf),
        ("Gradient Boosting", gb),
    ):

        probability = model.predict_proba(
            validation[feature_list]
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        model_rows.append({
            "Model": name,
            **{
                f"Validation_{key}": value
                for key, value in metrics(
                    y_validation,
                    prediction,
                    probability,
                ).items()
            },
        })

    model_rows.insert(
        0,
        {
            "Model": "Naive Baseline",
            "Validation_Accuracy": (
                y_validation
                .eq(majority)
                .mean()
            ),
            "Validation_ROC_AUC": 0.5,
        },
    )

    summary = pd.DataFrame(model_rows)

    return {
        "diagnostics_feature_target_correlation.csv":
            correlation_table,

        "diagnostics_feature_statistics.csv":
            statistics,

        "diagnostics_effect_sizes.csv":
            effects,

        "diagnostics_logistic_coefficients.csv":
            coefficients,

        "diagnostics_permutation_importance.csv":
            permutation_table,

        "diagnostics_gradient_boosting_importance.csv":
            gb_table,

        "diagnostics_distribution_shift.csv":
            distribution,

        "diagnostics_model_summary.csv":
            summary,
    }