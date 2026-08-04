# 14 - Hyperparameter Tuning with GridSearchCV

This project applies hyperparameter tuning to optimize a Random Forest classifier for predicting whether an employee belongs to the high salary group.

The model uses:

- Experience
- PerformanceScore
- Education
- Department

## GridSearchCV

GridSearchCV was used together with 5-fold Cross Validation to automatically search for the best combination of Random Forest hyperparameters.

Tested parameters:

- Number of trees (`n_estimators`)
- Maximum tree depth (`max_depth`)

## Results

Best parameters:

| Parameter | Value |
|---|---:|
| n_estimators | 200 |
| max_depth | None |

Performance:

| Metric | Score |
|---|---:|
| Cross Validation Accuracy | 54.2% |
| Test Accuracy | 54.2% |

## Conclusion

Hyperparameter tuning slightly improved Random Forest performance compared to the default configuration.

The results showed that GridSearchCV can find better model configurations, but the small dataset size limited the overall improvement.