# 07 - Linear Regression

This project applies Linear Regression to predict employee salary based on selected employee features.

The model uses:

- Experience
- PerformanceScore
- Education
- Department

Categorical variables were converted using one-hot encoding before training the model.

## Results

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| Original | 39648.67 | 73354.46 | -0.09 |
| Clean | 22190.16 | 26732.84 | -0.28 |

## Conclusion

The model confirmed the findings from exploratory analysis.

Removing salary outliers reduced prediction errors, but the low R² score indicates that the selected features were not sufficient to explain salary variability.