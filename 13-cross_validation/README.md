# 13 - Cross Validation Model Comparison

This project applies Cross Validation to compare multiple classification models for predicting whether an employee belongs to the high salary group.

The models use:

- Experience
- PerformanceScore
- Education
- Department

The evaluated models:

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- Decision Tree

## Cross Validation

Instead of using a single train/test split, 5-fold Cross Validation was applied.

The dataset was divided into multiple training and validation sets to measure model performance more reliably and reduce the impact of a single random split.

## Model Comparison

| Model | Mean Accuracy | Standard Deviation |
|---|---:|---:|
| SVM | 51.7% | 3.3% |
| KNN | 56.7% | 6.2% |
| Random Forest | 51.7% | 9.7% |
| Decision Tree | 52.5% | 5.7% |

KNN achieved the highest average accuracy, while SVM showed the most stable performance across different validation splits.

Random Forest achieved a higher score in one validation fold, but the larger standard deviation showed higher variability between splits.

## Conclusion

Cross Validation provided a more reliable evaluation of model performance compared to a single train/test split.

The results showed that KNN performed best on this dataset, but the overall accuracy remained limited due to the small dataset size and available features.