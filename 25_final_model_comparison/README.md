# 25 - Final Model Comparison

This project compares several classification models using multiple evaluation metrics to determine which model performs best on an imbalanced dataset.

The models use:

- Experience
- PerformanceScore
- ExperienceScore
- Education
- Department

## Model Comparison

The following models were compared:

- Logistic Regression
- Random Forest
- SVM
- Gradient Boosting

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Average Precision

## Results

| Model | Accuracy | Precision | Recall | F1 | Average Precision |
| ------------------- | --------: | ---------: | -----: | -----: | -----------------: |
| Logistic Regression | 64.3% | 16.7% | 100% | 28.6% | 50.0% |
| Random Forest | 92.9% | 0.0% | 0.0% | 0.0% | 33.3% |
| SVM | 92.9% | 0.0% | 0.0% | 0.0% | 25.0% |
| Gradient Boosting | 78.6% | 0.0% | 0.0% | 0.0% | 16.7% |

## Conclusion

Accuracy alone would suggest that Random Forest and SVM are the best models.

However, both models failed to identify the positive `True` class, resulting in 0% recall and F1 score.

Logistic Regression achieved 100% recall and the highest Average Precision of 50%, making it the most useful model for this particular classification task.

The experiment demonstrates why model selection should consider multiple evaluation metrics instead of relying only on accuracy.

Due to the small dataset size and very limited number of positive test examples, the results should be interpreted carefully.