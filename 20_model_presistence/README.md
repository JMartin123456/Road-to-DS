# 20 - Model Persistence

This project demonstrates how to save and load a trained machine learning model so it can be reused without retraining.

The model uses:

* Experience
* PerformanceScore
* Education
* Department

## Model Persistence

The trained Logistic Regression pipeline was saved using `joblib`.

The saved pipeline contains both:

* Data preprocessing
* Logistic Regression model

This allows the complete pipeline to be restored and used later.

## Results

The original and loaded models produced identical predictions and evaluation results.

| Metric            | Original | Loaded |
| ----------------- | -------: | -----: |
| Accuracy          |    64.3% |  64.3% |
| Average Precision |      50% |    50% |

## Conclusion

Model persistence allows trained models to be saved and reused without retraining.

Saving the complete pipeline also ensures that the same preprocessing steps are applied when the model is loaded and used again.
