"""Train and report the base random-forest model."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from market_pipeline import BASE_FEATURES, PROCESSED_DIR, metrics


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(train[BASE_FEATURES], train["Target"])
    for name in ("validation", "test"):
        frame = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
        probabilities = model.predict_proba(frame[BASE_FEATURES])[:, 1]
        print(name.title(), metrics(frame["Target"], (probabilities >= 0.5).astype(int), probabilities))
    print(pd.DataFrame({"Feature": BASE_FEATURES, "Importance": model.feature_importances_})
          .sort_values("Importance", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
