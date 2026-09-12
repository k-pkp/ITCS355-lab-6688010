"""Lab 2 — budgeted hyperparameter study.

Run:  python -m src.tune --trials 12 --budget-thb 150

The budget is enforced, not advisory. The study stops when projected spend would exceed
it, and reports what it did not get to. This is the habit the lab is teaching: compute is
a resource you spend deliberately, and a trial that is 0.3% better and four times the cost
is not better.

Every trial logs its estimated cost alongside its metric, so `scripts/compare_runs.py` can
rank by cost per point rather than by metric alone.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config, costs, data, seeds
from src.train import git_commit

# Five hyperparameters, chosen because each one changes a different thing about the model
# rather than being another way to say "bigger forest":
#   n_estimators     — how much averaging; trades wall-clock for variance reduction
#   max_depth        — how much each tree may memorise; None means grow until pure
#   min_samples_leaf — the floor on leaf support, the strongest overfitting brake here
#   max_features     — how decorrelated the trees are from each other
#   class_weight     — how much the 12% positive class is worth relative to the negatives,
#                      which moves the ranking this problem is actually scored on
SEARCH_SPACE: dict[str, list] = {
    "n_estimators": [200, 500],
    "max_depth": [6, 10, None],
    "min_samples_leaf": [1, 5, 20],
    "max_features": ["sqrt", 0.5],
    "class_weight": [None, "balanced"],
}


def grid(space: dict[str, list]) -> list[dict]:
    """Every combination in the space, in a stable order."""
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[key_name] for key_name in keys))]


def choose_trials(space: dict[str, list], trials: int, seed: int) -> list[dict]:
    """Pick `trials` configurations from the full grid, deterministically.

    Taking the first `trials` entries of the grid would hold the leading hyperparameters
    at their first value and vary only the trailing ones — a study that looks like five
    hyperparameters and is really two. Shuffling with a fixed seed first spreads the
    budget across every dimension and still reproduces exactly on a rerun.
    """
    combinations = grid(space)
    if trials >= len(combinations):
        return combinations

    shuffler = random.Random(seed)
    shuffled = list(combinations)
    shuffler.shuffle(shuffled)
    return shuffled[:trials]


def parse_args() -> argparse.Namespace:
    """Command line for the budgeted study."""
    parser = argparse.ArgumentParser(description="ITCS355 Lab 2 — budgeted study")
    parser.add_argument("--trials", type=int, default=12, help="minimum 12 for the lab")
    parser.add_argument("--budget-thb", type=float, default=150.0)
    parser.add_argument("--instance", default="local", help="key into src/costs.py PRICE_TABLE")
    parser.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    parser.add_argument("--experiment", default="itcs355-lab2")
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/tune_checkpoint.json"),
                   help="Resume file. Spot interruption should cost minutes, not the run.")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    """Read the resume file, or start from an empty study.

        An interrupted spot instance should cost minutes, not the whole run, which is the
        only reason this file exists.
        """
    if path.exists():
        return json.loads(path.read_text())
    return {"completed": [], "spent_thb": 0.0}


def save_checkpoint(path: Path, state: dict) -> None:
    """Write the resume file after every completed trial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def main() -> None:
    """Run trials until the budget is spent, reporting what could not be afforded."""
    options = parse_args()
    loaded_config = config.load(strict=False)
    seed = seeds.set_all(options.seed)

    frame = data.load_raw(loaded_config.raw_path)
    fingerprint = data.data_fingerprint(loaded_config.raw_path)
    train_df, val_df, test_df = data.split(frame, seed=seed)

    mlflow.set_tracking_uri(loaded_config.mlflow_tracking_uri)
    mlflow.set_experiment(options.experiment)

    state = load_checkpoint(options.checkpoint)
    candidates = choose_trials(SEARCH_SPACE, options.trials, seed)
    rate = costs.hourly_rate(loaded_config.provider, options.instance)

    skipped: list[dict] = []
    for trial_index, params in enumerate(candidates):
        key = json.dumps(params, sort_keys=True)
        if key in state["completed"]:
            print(f"trial {trial_index}: already done, skipping (resumed from checkpoint)")
            continue

        if state["spent_thb"] >= options.budget_thb:
            skipped.append(params)
            continue

        started = time.perf_counter()
        with mlflow.start_run(run_name=f"trial-{trial_index:02d}"):
            model = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
            model.fit(train_df[data.FEATURES], train_df[data.TARGET])

            metrics = {}
            for name, part in (("val", val_df), ("test", test_df)):
                proba = model.predict_proba(part[data.FEATURES])[:, 1]
                metrics[f"{name}_roc_auc"] = float(roc_auc_score(part[data.TARGET], proba))
                metrics[f"{name}_pr_auc"] = float(average_precision_score(part[data.TARGET], proba))

            elapsed_h = (time.perf_counter() - started) / 3600.0
            trial_cost = elapsed_h * rate
            state["spent_thb"] += trial_cost

            mlflow.log_params({**params, "seed": seed, "instance": options.instance})
            mlflow.log_metrics({
                **metrics,
                "duration_s": round(elapsed_h * 3600, 3),
                "cost_thb": round(trial_cost, 4),
            })
            mlflow.set_tags({
                "git_commit": git_commit(),
                "data_fingerprint": fingerprint,
                "lab": "2",
            })
            mlflow.sklearn.log_model(model, name="model")

        state["completed"].append(key)
        save_checkpoint(options.checkpoint, state)
        print(f"trial {trial_index}: {params} -> val_roc_auc={metrics['val_roc_auc']:.4f} "
              f"cost={trial_cost:.4f} THB  cumulative={state['spent_thb']:.4f}")

    print(f"\nspent {state['spent_thb']:.4f} of {options.budget_thb} THB")
    if skipped:
        print(f"BUDGET EXHAUSTED — {len(skipped)} configurations not run:")
        for skipped_configuration in skipped:
            print(f"  {skipped_configuration}")
        print("Report this in your README. Which trials you could not afford is a finding, "
              "not an embarrassment.")


if __name__ == "__main__":
    main()
