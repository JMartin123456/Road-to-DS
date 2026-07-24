# 09 - Decision Tree Classifier

This project applies Decision Tree classification to predict whether an employee belongs to the high salary group.

The model uses:

- Experience
- PerformanceScore
- Education
- Department

## Results

The best performing model achieved:

Accuracy: 60.9%

Feature importance showed that the model relied mainly on:

1. PerformanceScore
2. Experience

## Tree Depth Experiment

| Depth | Train Accuracy | Test Accuracy |
|---|---:|---:|
| 2 | 65.2% | 30.4% |
| 3 | 68.5% | 30.4% |
| 5 | 75.0% | 39.1% |
| Unlimited | 100% | 60.9% |

The experiment showed the effect of tree complexity:

- Smaller trees can underfit the data.
- Large trees can memorize training data and cause overfitting.

## Conclusion

Decision Trees were able to find some patterns in the data, but the available features were still not enough for reliable salary classification.