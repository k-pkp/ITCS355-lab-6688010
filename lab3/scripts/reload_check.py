"""Lab 2 — prove the registered model can be reloaded by version, from the registry.

    python scripts/reload_check.py --name itcs355-<studentid> --version 3

This is the lab's quiet test. Models that cannot be reloaded six months later are the
commonest form of dead work in industry, and the cause is nearly always a serialization
assumption: a custom class that no longer exists, a library version that moved, a
preprocessing step that only ever lived in a notebook.

Loading from a local file instead of the registry defeats the purpose and is checked.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from src import config, data


def main() -> int:
    """Load the registered model by version, from the registry, and score held-out rows."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--name", required=True, help="registered model name")
    argument_parser.add_argument("--version", required=True)
    argument_parser.add_argument("--rows", type=int, default=5)
    options = argument_parser.parse_args()

    loaded_config = config.load(strict=False)
    mlflow.set_tracking_uri(loaded_config.mlflow_tracking_uri)

    uri = f"models:/{options.name}/{options.version}"
    print(f"loading {uri}")
    model = mlflow.sklearn.load_model(uri)

    frame = data.load_raw(loaded_config.raw_path)
    _, _, test_df = data.split(frame, seed=20260101)
    sample = test_df.head(options.rows)
    preds = model.predict_proba(sample[data.FEATURES])[:, 1]

    for rid, probability in zip(sample[data.IDENTIFIER], preds):
        print(f"  reading {rid}: p(failure)={probability:.4f}")
    print("\nPASS  model reloaded from the registry and scored rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
