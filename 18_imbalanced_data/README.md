# 18 - Handling Imbalanced Data

This project demonstrates how imbalanced datasets affect classification models and why accuracy alone is often not a reliable evaluation metric.

The model uses:

- Experience
- PerformanceScore
- Education
- Department

## Imbalanced Dataset

The original balanced dataset was modified to create an imbalanced classification problem.

This simulates real-world scenarios where one class is much less frequent than the other, such as fraud detection or disease diagnosis.

## Class Weight

Two Logistic Regression models were compared:

- Default Logistic Regression
- Logistic Regression with `class_weight="balanced"`

The balanced version increases the importance of the minority class during training.

## Results

Without class weighting:

- High accuracy
- Failed to detect the minority class
- Recall = 0%

With `class_weight="balanced"`:

- Lower overall accuracy
- Successfully detected the minority class
- Significantly improved recall

## Conclusion

This experiment shows that accuracy can be misleading when working with imbalanced datasets.

Using `class_weight="balanced"` helps the model pay more attention to the minority class, making recall a much more meaningful metric than accuracy in these situations.