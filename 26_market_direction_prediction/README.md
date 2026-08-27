# EUR/USD market direction prediction

This workflow predicts whether the next daily EUR/USD close is higher than
the current close.  It uses chronological train/validation/test splits (70%,
15%, 15%) and keeps the existing CSV and figure output names.

## Run

Run from the repository root:

```powershell
python 26_market_direction_prediction\src\26_feature_engineering.py
python 26_market_direction_prediction\src\26_dataset_preparation.py
python 26_market_direction_prediction\src\26_feature_engineering_advanced.py
python 26_market_direction_prediction\src\26_dataset_preparation_advanced.py
python 26_market_direction_prediction\src\26_model_comparison.py
python 26_market_direction_prediction\src\26_model_comparison_advanced.py
```

`26_data_collection.py` downloads fresh data. Set
`ALPHAVANTAGE_API_KEY` first; placeholder keys are rejected.

Feature selection, tuning, diagnostics, and threshold analysis are separate
scripts in `src`. Shared implementation lives in `src\market_pipeline.py`;
scripts expose `main()` guards and can therefore be imported safely.

## Important fixes

- The final row's unknown next-day target is now missing and removed during
  preparation, rather than incorrectly labelled as a down day.
- All divisions and rolling calculations handle zero/insufficient history
  without infinite values.
- Advanced preparation no longer assumes exactly 5,000 rows and validates
  every feature/target consistently.
- Logistic regression is scaled, metric calculations handle missing classes,
  and threshold selection uses validation only.
- Repeated loading, feature lists, model evaluation, and output logic were
  consolidated to remove drift between the basic and advanced scripts.
