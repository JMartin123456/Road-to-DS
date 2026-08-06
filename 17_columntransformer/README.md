# 17 - Complete Machine Learning Pipeline

This project builds a complete Machine Learning pipeline using ColumnTransformer and Pipeline to automate data preprocessing and model training.

The model uses:

- Experience
- PerformanceScore
- Education
- Department

## ColumnTransformer

ColumnTransformer was used to apply different preprocessing steps to different feature types.

- Numerical features were standardized using StandardScaler.
- Categorical features were encoded using OneHotEncoder.

This allowed the model to receive raw data without manual preprocessing.

## Pipeline

Pipeline combined the preprocessing steps and the Logistic Regression model into a single workflow.

The pipeline automatically:

- scales numerical features
- encodes categorical features
- trains the model
- applies the same preprocessing during prediction

## Results

Performance:

| Metric | Score |
|---|---:|
| Accuracy | 50.0% |

## Conclusion

Using Pipeline and ColumnTransformer simplifies the machine learning workflow by combining preprocessing and model training into a single reusable object.

Although the model achieved similar performance to previous Logistic Regression experiments, the implementation is more robust and closer to real-world machine learning projects.