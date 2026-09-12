"""Training entry point.

Run locally:      python -m src.train --n-estimators 200 --max-depth 8
Run in Docker:    make reproduce

Every run logs: all hyperparameters, the seed, validation AND test metrics separately,
the data fingerprint, and the Git commit. A metric that cannot be traced to code and
data is not evidence of anything.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config, data, seeds


def git_commit() -> str:
    """Return the commit this run was produced from, or "unknown" if it cannot be found.

    The container has no .git directory and no git binary, so inside the image the
    subprocess call cannot work. `make reproduce` passes the commit in as GIT_COMMIT
    instead; without that fallback every containerised run logs "unknown" and the
    lineage chain the labs are built around is broken at its first link.
    """
    from_environment = os.environ.get("GIT_COMMIT", "").strip()
    if from_environment:
        return from_environment

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=config.REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    """Command line for a training run. Every flag here is logged as a parameter."""
    parser = argparse.ArgumentParser(description="ITCS355 Lab 1 — reproducible training")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    parser.add_argument("--experiment", default="itcs355-lab1")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--metrics-out", type=Path, default=None,
                   help="Write final metrics as JSON. Used by `make verify`.")
    return parser.parse_args()


def main() -> None:
    """Train once, log everything needed to trace the metric back to code and data."""
    options = parse_args()
    loaded_config = config.load(strict=False)
    seed = seeds.set_all(options.seed)

    frame = data.load_raw(loaded_config.raw_path)
    fingerprint = data.data_fingerprint(loaded_config.raw_path)
    train_df, val_df, test_df = data.split(frame, seed=seed)

    mlflow.set_tracking_uri(loaded_config.mlflow_tracking_uri)
    mlflow.set_experiment(options.experiment)

    with mlflow.start_run(run_name=options.run_name):
        mlflow.log_params({
            "n_estimators": options.n_estimators,
            "max_depth": options.max_depth,
            "min_samples_leaf": options.min_samples_leaf,
            "seed": seed,
            "n_features": len(data.FEATURES),
        })
        # Provenance. This is what makes the metric traceable.
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_fingerprint": fingerprint,
            "split_strategy": "group_by_machine_id",
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_test_rows": len(test_df),
        })

        model = RandomForestClassifier(
            n_estimators=options.n_estimators,
            max_depth=options.max_depth,
            min_samples_leaf=options.min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])

        metrics: dict[str, float] = {}
        for name, part in (("val", val_df), ("test", test_df)):
            proba = model.predict_proba(part[data.FEATURES])[:, 1]
            metrics[f"{name}_roc_auc"] = float(roc_auc_score(part[data.TARGET], proba))
            metrics[f"{name}_pr_auc"] = float(average_precision_score(part[data.TARGET], proba))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        print(json.dumps({"seed": seed, "data_fingerprint": fingerprint, **metrics}, indent=2))
        if options.metrics_out:
            options.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            options.metrics_out.write_text(json.dumps(
                {"seed": seed, "data_fingerprint": fingerprint, **metrics}, indent=2))


if __name__ == "__main__":
    main()
