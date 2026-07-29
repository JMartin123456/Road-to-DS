# 11 - K-Nearest Neighbors Classifier

This project applies the K-Nearest Neighbors (KNN) classifier to predict whether an employee belongs to the high salary group.

The model uses:

- Experience
- PerformanceScore
- Education
- Department

## Feature Scaling

Since KNN is based on distance calculations, feature scaling was applied using StandardScaler.

Without scaling, features with larger numerical ranges could dominate the distance calculation. Scaling allowed all features to contribute more equally to the similarity calculation.

## K Experiment

The model was tested with different values of K (number of nearest neighbors).

| K | Accuracy |
|---:|---:|
| 1 | 70.8% |
| 3 | 58.3% |
| 5 | 50.0% |
| 7 | 41.7% |
| 9 | 50.0% |

The best result was achieved with K=1. However, additional tests with different train/test splits showed that KNN performance was sensitive to the dataset division.

## Conclusion

Feature scaling significantly improved KNN performance because the algorithm relies on distance calculations.

The results showed that scaling allowed the model to better compare employee similarities, but the small dataset size caused variation between different train/test splits.