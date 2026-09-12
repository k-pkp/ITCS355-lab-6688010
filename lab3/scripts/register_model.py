"""Lab 2 — register the chosen model with lineage back to exact code and exact data.

    python scripts/register_model.py --stage Staging

Registers the chosen configuration in the MLflow Model Registry and stamps the eight
lineage fields the lab requires ON THE VERSION, not on the run. Tags set on the run are
a common near-miss: the run and the registered version are different objects, and it is
the version a future engineer starts from.

Lineage exists to answer one question — six months from now, can you rebuild this exact
model? Every missing field below is a path by which the answer becomes no.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config, data, seeds
from src.train import git_commit

# The configuration chosen in reports/lab2-comparison.md. It is not the highest-scoring
# trial; it is the cheapest one whose score is inside the seed noise of the best.
CHOSEN_HYPERPARAMETERS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": None,
}

# Lab 3 Task 4 needs a canary that is worse by a small margin, not obviously broken. This
# one is the shape of a real regression: somebody removed the depth limit and switched on
# class weighting to "help with the class imbalance". It still returns sensible-looking
# probabilities and it still trains cleanly; it just ranks 0.016 AUC worse on the same
# split (test 0.8341 against 0.8499). Obviously broken versions make the exercise trivial
# — the hard part of a canary is noticing a degradation that does not announce itself.
#
# Shrinking the forest was tried first and rejected: 40 trees at depth 3 scored *better*
# than the registered model on this data, which is its own lesson about how little the
# headline metric moves on a problem this easy.
DEGRADED_HYPERPARAMETERS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
}

VARIANTS = {"chosen": CHOSEN_HYPERPARAMETERS, "degraded": DEGRADED_HYPERPARAMETERS}

REPO_ROOT = Path(__file__).resolve().parents[1]
DVC_POINTER = REPO_ROOT / "data" / "raw.dvc"
IMAGE_DIGEST_FILE = REPO_ROOT / "reports" / "image-digest.txt"


def read_data_version() -> str:
    """Return the DVC content hash of data/raw, or "unversioned" if DVC is not set up.

    This is the data half of lineage. A metric traceable to code but not to data tells
    you which program produced a number and nothing about what it was fed.
    """
    if not DVC_POINTER.exists():
        return "unversioned"

    pointer = yaml.safe_load(DVC_POINTER.read_text())
    outputs = pointer.get("outs", [])
    if not outputs:
        return "unversioned"
    return str(outputs[0].get("md5", "unversioned"))


def read_image_digest() -> str:
    """Return the digest of the pushed training image, or "not-pushed".

    `make image-push` writes this file. The digest, not the tag, is what pins the
    environment the model was trained in — a tag can be moved onto other bytes.
    """
    if not IMAGE_DIGEST_FILE.exists():
        return "not-pushed"
    return IMAGE_DIGEST_FILE.read_text().strip().splitlines()[-1]


def training_job_identifier(managed_job_id: str | None) -> str:
    """Name the compute that produced this model.

    A managed training job supplies its own id. This study ran on the development
    machine, so the identifier says so explicitly rather than leaving the field blank
    or, worse, implying a job that never existed.
    """
    if managed_job_id:
        return managed_job_id
    return f"local://{socket.gethostname()}/{time.strftime('%Y%m%dT%H%M%S')}"


def parse_command_line() -> argparse.Namespace:
    """Command line for registration."""
    parser = argparse.ArgumentParser(description="ITCS355 Lab 2 — register with lineage")
    parser.add_argument("--name", default=None, help="defaults to MODEL_REGISTRY_NAME")
    parser.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    parser.add_argument("--experiment", default="itcs355-lab2")
    parser.add_argument("--stage", default="Staging",
                        help="registry alias to promote the new version to")
    parser.add_argument("--training-job-id", default=None,
                        help="managed job id, when training ran on managed compute")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="chosen",
                        help="'degraded' registers the deliberately worse Lab 3 canary")
    return parser.parse_args()


def main() -> int:
    """Train the chosen configuration, register it, and stamp its lineage."""
    options = parse_command_line()
    loaded_config = config.load(strict=False)
    registered_name = options.name or loaded_config.model_registry_name

    seed = seeds.set_all(options.seed)
    frame = data.load_raw(loaded_config.raw_path)
    fingerprint = data.data_fingerprint(loaded_config.raw_path)
    train_frame, validation_frame, test_frame = data.split(frame, seed=seed)

    mlflow.set_tracking_uri(loaded_config.mlflow_tracking_uri)
    mlflow.set_experiment(options.experiment)

    hyperparameters = VARIANTS[options.variant]

    with mlflow.start_run(run_name=f"{options.variant}-for-registration") as active_run:
        model = RandomForestClassifier(random_state=seed, n_jobs=-1, **hyperparameters)
        model.fit(train_frame[data.FEATURES], train_frame[data.TARGET])

        metrics: dict[str, float] = {}
        for split_name, split_frame in (("val", validation_frame), ("test", test_frame)):
            probabilities = model.predict_proba(split_frame[data.FEATURES])[:, 1]
            metrics[f"{split_name}_roc_auc"] = float(
                roc_auc_score(split_frame[data.TARGET], probabilities))
            metrics[f"{split_name}_pr_auc"] = float(
                average_precision_score(split_frame[data.TARGET], probabilities))

        mlflow.log_params({**hyperparameters, "seed": seed, "variant": options.variant})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_fingerprint": fingerprint,
            "lab": "2",
            "variant": options.variant,
        })
        model_info = mlflow.sklearn.log_model(model, name="model")
        run_id = active_run.info.run_id

    client = MlflowClient()
    version = mlflow.register_model(model_uri=model_info.model_uri, name=registered_name)

    lineage = {
        "git_commit": git_commit(),
        "data_version": read_data_version(),
        "mlflow_run_id": run_id,
        "training_job_id": training_job_identifier(options.training_job_id),
        "image_digest": read_image_digest(),
        "seed": str(seed),
        "metric_val": f"{metrics['val_roc_auc']:.6f}",
        "metric_test": f"{metrics['test_roc_auc']:.6f}",
    }
    for tag_name, tag_value in lineage.items():
        client.set_model_version_tag(registered_name, version.version, tag_name, tag_value)

    # Promotion. Aliases replaced stages in MLflow 3, and an alias is the better model of
    # the idea anyway: "staging" is a pointer somebody moves, not a property of the version.
    client.set_registered_model_alias(registered_name, options.stage.lower(), version.version)

    print(f"registered {registered_name} version {version.version}")
    print(f"promoted to alias '{options.stage.lower()}'")
    print(json.dumps(lineage, indent=2))

    missing_fields = [name for name, value in lineage.items()
                      if value in ("unversioned", "not-pushed", "unknown")]
    if missing_fields:
        print(f"\nWARNING incomplete lineage: {missing_fields}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
