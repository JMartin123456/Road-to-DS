"""Train and report the base gradient-boosting model."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from market_pipeline import BASE_FEATURES, PROCESSED_DIR, metrics


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    model.fit(train[BASE_FEATURES], train["Target"])
    for name in ("validation", "test"):
        frame = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
        probabilities = model.predict_proba(frame[BASE_FEATURES])[:, 1]
        print(name.title(), metrics(frame["Target"], (probabilities >= 0.5).astype(int), probabilities))
    print(pd.DataFrame({"Feature": BASE_FEATURES, "Importance": model.feature_importances_})
          .sort_values("Importance", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
