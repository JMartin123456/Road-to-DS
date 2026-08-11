# 23 - Ensemble Learning

This project demonstrates how multiple classification models can be combined using ensemble learning techniques.

The models used in the ensemble are:

* Logistic Regression
* Random Forest
* Support Vector Machine

The ensemble methods tested were:

* Hard Voting
* Soft Voting
* Stacking

## Hard Voting

Hard Voting combines the predictions of multiple models and selects the class with the majority of votes.

For example:

```text
Logistic Regression → True
Random Forest       → False
SVM                 → False

Final prediction → False
```

In this experiment, Random Forest and SVM consistently predicted the majority class, causing Hard Voting to also predict mostly `False`.

## Soft Voting

Soft Voting combines the predicted probabilities of the individual models instead of only using their final predictions.

The probability of each class is averaged across the models and the class with the higher combined probability is selected.

The experiment showed that Soft Voting produced results very similar to Hard Voting.

## Stacking

Stacking uses the predictions of several base models as input for another model called the meta-model.

The structure is:

```text
Logistic Regression ─┐
Random Forest       ─┼──→ Meta-model → Final prediction
SVM                 ─┘
```

Logistic Regression was used as the final meta-model.

## Results

The models were evaluated on the same imbalanced dataset.

| Model       | Accuracy |
| ----------- | -------: |
| Hard Voting |   92.86% |
| Soft Voting |   92.86% |
| Stacking    |   57.14% |

The confusion matrices showed that Hard Voting and Soft Voting predicted all test samples as the majority class.

Stacking predicted some positive cases, but none of them were true positives.

## Conclusion

The experiment demonstrated that ensemble learning does not automatically produce better results.

Hard Voting and Soft Voting achieved high accuracy, but they failed to detect the minority class. This shows why accuracy can be misleading when working with imbalanced datasets.

Stacking produced lower accuracy and several false positive predictions, but it also demonstrated a different decision pattern from the voting approaches.

Because the dataset is small and highly imbalanced, the results should not be considered a reliable measure of real-world model performance.

The main purpose of this experiment was to understand how Hard Voting, Soft Voting and Stacking combine multiple models and how their behaviour can differ on the same dataset.
