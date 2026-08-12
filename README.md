# Road to Data Science

This repository documents my journey learning Python, Data Analysis and Machine Learning.

The projects follow a step-by-step progression from Python fundamentals and data analysis to machine learning models, model evaluation, optimization techniques and model interpretation.

## Completed Projects

### 01 - Python Refresh
- functions
- dictionaries
- sorting
- lambda expressions

### 02 - Pandas Basics
- CSV loading
- filtering
- groupby
- aggregations

### 03 - Data Cleaning
- missing values
- duplicates
- data type handling

### 04 - Data Visualization
- matplotlib
- histograms
- bar charts
- scatter plots

### 05 - NumPy Statistics
- statistics
- percentiles
- variance and standard deviation
- outlier detection

### 06 - Exploratory Data Analysis (EDA)
- data exploration workflow
- salary analysis
- group comparisons
- mean vs median analysis
- correlation analysis
- heatmaps
- scatter plots
- categorical distributions
- outlier impact analysis
- IQR-based outlier filtering

### 07 - Linear Regression
- feature selection
- categorical encoding
- train/test split
- linear regression
- model evaluation
- coefficient interpretation

### 08 - Logistic Regression
- binary classification
- train/test split
- logistic regression
- accuracy
- confusion matrix
- classification report

### 09 - Decision Tree Classifier
- decision tree classification
- feature importance
- tree depth tuning
- overfitting and underfitting
- model evaluation

### 10 - K-Nearest Neighbors Classifier
- KNN classification
- distance-based learning
- feature scaling
- K parameter comparison
- model evaluation

### 11 - Support Vector Machine Classifier
- SVM classification
- feature scaling
- kernel comparison
- linear, RBF and polynomial kernels
- model evaluation

### 12 - Model Evaluation and Comparison
- comparing classification models
- accuracy comparison
- confusion matrix analysis
- classification metrics

### 13 - Cross Validation
- K-fold cross validation
- cross validation scores
- model stability evaluation
- reducing dependency on single train/test split

### 14 - Hyperparameter Tuning with GridSearchCV
- GridSearchCV
- hyperparameter optimization
- Random Forest tuning
- cross validation during parameter search

### 15 - Feature Importance Analysis
- Random Forest feature importance
- permutation importance
- model interpretation
- comparing internal and external feature importance

### 16 - ROC Curve and AUC Evaluation
- ROC curve
- AUC score
- threshold evaluation
- probability-based classification evaluation
- Logistic Regression performance analysis

### 17 - Complete Machine Learning Pipeline
- ColumnTransformer
- Pipeline
- StandardScaler
- OneHotEncoder
- automated preprocessing
- Logistic Regression

### 18 - Handling Imbalanced Data
- imbalanced datasets
- class_weight
- minority class detection
- accuracy vs recall
- Logistic Regression

### 19 - Precision-Recall Curve
- precision and recall trade-off
- Precision-Recall Curve
- Average Precision Score
- evaluating imbalanced classification models

### 20 - Model Persistence

* saving trained models with `joblib`
* loading saved models
* model reuse without retraining
* saving complete ML pipelines
* verifying identical predictions after loading

### 21 - Feature Engineering

* creating new features from existing data
* feature interaction
* `ExperienceScore`
* evaluating feature impact on model performance
* comparing model performance before and after feature engineering

### 22 - Logistic Regression Interpretation
* Logistic Regression coefficients
* coefficient direction
* absolute coefficient values
* Odds Ratio
* categorical reference categories
* interpreting model relationships
* understanding association vs causation

### 23 - Ensemble Learning

* ensemble learning
* Hard Voting
* Soft Voting
* Stacking
* combining multiple classification models
* comparing ensemble approaches
* limitations of ensemble methods on small and imbalanced datasets
* accuracy vs minority class performance

### 24 - Gradient Boosting

- Gradient Boosting classification
- sequential ensemble learning
- `GradientBoostingClassifier`
- hyperparameter tuning with `GridSearchCV`
- `n_estimators`
- `learning_rate`
- `max_depth`
- F1-based model optimization
- comparing accuracy and minority class performance
- limitations of Gradient Boosting on highly imbalanced data

### 25 - Final Model Comparison

- comparing Logistic Regression, Random Forest, SVM and Gradient Boosting
- Accuracy
- Precision
- Recall
- F1 Score
- Average Precision
- model comparison on an imbalanced dataset
- understanding why accuracy alone can be misleading
- selecting a model based on the requirements of the problem

## Final Model Comparison

The final project compared several classification models:

- Logistic Regression
- Random Forest
- SVM
- Gradient Boosting

The models were evaluated using multiple metrics rather than accuracy alone.

Logistic Regression achieved the highest Recall and Average Precision on the test set, while Random Forest and SVM achieved higher accuracy but failed to detect the positive class.

This final comparison demonstrated the importance of choosing evaluation metrics based on the actual problem rather than selecting a model based only on accuracy.

## Final Takeaways

Throughout the project I learned how to:

- prepare and clean datasets
- perform exploratory data analysis
- build regression and classification models
- preprocess numerical and categorical features
- use pipelines and ColumnTransformer
- evaluate models using multiple metrics
- work with imbalanced datasets
- perform cross validation
- tune hyperparameters with GridSearchCV
- interpret Logistic Regression coefficients and Odds Ratios
- analyze feature importance
- use Precision-Recall and ROC curves
- save and load trained models
- compare multiple machine learning models
- select models based on the requirements of the problem

The project progressed from basic Python and data analysis to building, evaluating, interpreting and comparing complete machine learning workflows.
