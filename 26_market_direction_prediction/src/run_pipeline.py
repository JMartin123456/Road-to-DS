from __future__ import annotations

from datasets import load_splits
from features import BASE_FEATURES
from models import compare_models
from analysis import diagnostics


def main() -> None:
    # 1. Načítanie datasetov
    train, validation, test = load_splits()

    # 2. Porovnanie základných modelov
    validation_result, test_result, _, _ = compare_models(
        train,
        validation,
        test,
        BASE_FEATURES,
    )

    print("=== VALIDATION ===")
    print(validation_result.to_string(index=False))

    print("\n=== TEST ===")
    print(test_result.to_string(index=False))

    # 3. Diagnostika
    tables = diagnostics(
        train,
        validation,
        test,
        BASE_FEATURES,
    )

    print(f"\nCreated {len(tables)} diagnostic tables.")

    # Zatiaľ ich iba vypíšeme / sprístupníme v pamäti.
    for name, table in tables.items():
        print(f"- {name}: {len(table)} rows")


if __name__ == "__main__":
    main()