# 10 - Random Forest Classifier

This project applies Random Forest classification to predict whether an employee belongs to the high salary group.

The model uses:

- Experience
- PerformanceScore
- Education
- Department

## Results

The Random Forest model was evaluated using different numbers of decision trees.

| Number of Trees | Accuracy |
|---|---:|
| 10 | 41.7% |
| 200 | 54.2% |

Increasing the number of trees improved the model performance, but the Random Forest model did not outperform the Decision Tree classifier.

## Feature Importance

The model identified the most important features:

1. PerformanceScore
2. Experience

Other features such as education and department had a smaller influence on predictions.

## Conclusion

Random Forest was able to find similar patterns as the Decision Tree model. PerformanceScore and Experience were consistently the most important variables across multiple machine learning approaches.

However, the model performance was still limited, which suggests that the available features do not provide enough information for reliable salary classification.