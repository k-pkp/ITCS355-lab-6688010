"""Export a trained model to a local file, for the CI integration test.

Real deployments load a registered version from the registry. This exists only so CI can
start the service without a registry, and it is not acceptable in a submitted Lab 3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.ensemble import RandomForestClassifier

from src import config, data, seeds


def main() -> int:
    """Train a small model and write it to a file, so CI can start the service without a registry."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--out", type=Path, default=Path("reports/model.joblib"))
    argument_parser.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    options = argument_parser.parse_args()

    loaded_config = config.load(strict=False)
    seed = seeds.set_all(options.seed)
    frame = data.load_raw(loaded_config.raw_path)
    train_df, _, _ = data.split(frame, seed=seed)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   random_state=seed, n_jobs=-1)
    model.fit(train_df[data.FEATURES], train_df[data.TARGET])
    options.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, options.out)
    print(f"wrote {options.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
