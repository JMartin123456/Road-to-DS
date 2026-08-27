"""Train the scaled logistic-regression baseline."""

import pandas as pd

from market_pipeline import BASE_FEATURES, PROCESSED_DIR, metrics, model_factories


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    model = model_factories()["Logistic Regression"]
    model.fit(train[BASE_FEATURES], train["Target"])
    for name, frame in (("Validation", validation), ("Test", test)):
        probabilities = model.predict_proba(frame[BASE_FEATURES])[:, 1]
        print(name, metrics(frame["Target"], (probabilities >= 0.5).astype(int), probabilities))
    coefficients = pd.DataFrame({"Feature": BASE_FEATURES, "Coefficient": model[-1].coef_[0]})
    coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()
    print(coefficients.sort_values("Absolute_Coefficient", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
