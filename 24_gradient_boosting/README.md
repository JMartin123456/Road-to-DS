# 24 - Gradient Boosting

This project demonstrates how Gradient Boosting can be used for classification and how its hyperparameters can be optimized using `GridSearchCV`.

The model uses:

- Experience
- PerformanceScore
- ExperienceScore
- Education
- Department

## Gradient Boosting

A `GradientBoostingClassifier` was trained using:

- ColumnTransformer
- StandardScaler
- OneHotEncoder
- Pipeline

Gradient Boosting builds models sequentially, where each new model attempts to improve the errors made by previous models.

## Hyperparameter Tuning

The model was optimized using `GridSearchCV` with:

- `n_estimators`
- `learning_rate`
- `max_depth`
- 5-fold cross validation
- F1 score

## Results

| Model | Accuracy | True Recall | True F1 |
| ----------------------- | --------: | ----------: | ------: |
| Gradient Boosting | 78.6% | 0% | 0 |
| Tuned Gradient Boosting | 92.9% | 0% | 0 |

Best parameters:

| Parameter | Value |
| ---------------- | ----: |
| `learning_rate` | 0.05 |
| `max_depth` | 1 |
| `n_estimators` | 50 |

Best cross-validation F1:

```text
0.2333