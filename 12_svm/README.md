# 12 - Support Vector Machine Classifier

This project applies Support Vector Machine (SVM) classification to predict whether an employee belongs to the high salary group.

The model uses:

- Experience
- PerformanceScore
- Education
- Department

## Feature Scaling

Since SVM is sensitive to feature ranges, StandardScaler was applied before training the model.

Scaling helps ensure that features with larger values do not dominate the decision boundary.

## Kernel Experiment

Different SVM kernels were tested to compare their effect on classification performance.

| Kernel | Accuracy |
|---|---:|
| Linear | 33.3% |
| RBF | 54.2% |
| Polynomial | 54.2% |

The linear kernel performed worse because the data could not be separated effectively using a simple decision boundary.

RBF and polynomial kernels achieved better results by allowing more complex decision boundaries.

## Conclusion

SVM was able to find some patterns in the employee data, but the overall performance remained limited.

The results suggest that the selected features do not contain enough information for reliable salary classification.