# 21 - Feature Engineering

This project demonstrates how Feature Engineering can be used to create new features from existing data and evaluate whether they improve model performance.

The model uses:

* Experience
* PerformanceScore
* Education
* Department

## Feature Engineering

A new feature was created by combining `Experience` and `PerformanceScore`:

```python
ExperienceScore = Experience * PerformanceScore
```

The new feature was added to the preprocessing pipeline and used by the Logistic Regression model.

## Results

The model performance remained unchanged after adding the new feature.

| Metric            | Score |
| ----------------- | ----: |
| Accuracy          | 64.3% |
| Average Precision |   50% |

## Conclusion

The new `ExperienceScore` feature did not improve model performance on this dataset.

This demonstrates that Feature Engineering does not automatically improve a model and that new features should be evaluated based on their actual impact on model performance.
