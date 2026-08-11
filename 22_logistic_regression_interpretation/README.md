# 22 - Logistic Regression Interpretation

This project demonstrates how Logistic Regression coefficients can be interpreted to understand how features influence model predictions.

The model uses:

* Experience
* PerformanceScore
* ExperienceScore
* Education
* Department

## Model Interpretation

The Logistic Regression coefficients were analyzed using:

* model coefficients
* coefficient direction
* absolute coefficient values
* Odds Ratio
* categorical reference categories

Positive coefficients move the prediction towards `HighSalary=True`, while negative coefficients move it towards `HighSalary=False`.

## Results

The strongest coefficients in the model included:

| Feature               | Coefficient | Odds Ratio |
| --------------------- | ----------: | ---------: |
| Department_IT         |       0.742 |       2.10 |
| Education_High School |      -0.640 |       0.53 |
| Department_Operations |      -0.633 |       0.53 |
| Department_Sales      |       0.632 |       1.88 |

The reference categories were determined from the fitted `OneHotEncoder` using `drop="first"`.

## Conclusion

Logistic Regression coefficients can provide insight into how individual features influence model predictions.

Odds Ratio provides a more interpretable way to understand the effect of coefficients, while categorical features must be interpreted relative to their reference category.

The analysis also demonstrates that model relationships should not automatically be interpreted as causal relationships.
