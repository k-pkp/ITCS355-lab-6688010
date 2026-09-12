"""Lab 5 — the registration gate.

Compares a candidate against the currently registered production model. Exits non-zero
when the candidate should NOT be registered, which is what makes the pipeline condition
real rather than decorative.

    python scripts/evaluation_gate.py --metrics reports/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Both numbers are measured, not chosen for looking reasonable. Working in
# reports/lab5-cost.md and reports/lab2-comparison.md.
#
# MIN_ABSOLUTE — the floor below which this system should not exist. Ranking the same test
# split by a single raw sensor reading scores: vibration_mm_s 0.8257, hours_since_service
# 0.7674, temp_c 0.7285. A model that cannot beat "sort by vibration" is worse than a
# spreadsheet, and everything downstream of it — the registry, the endpoint, the drift
# monitor, the bill — is pure cost. So the floor sits just above the best heuristic rather
# than at the scaffold's 0.75, which would have passed a model that loses to one column.
MIN_ABSOLUTE = 0.83

# MIN_IMPROVEMENT — the margin below which "better" is unprovable. Test ROC AUC for the
# registered configuration varies by sd 0.0087 across five seeds (0.8540 mean). An
# improvement smaller than that is a different random split, not a better model, and
# promoting it burns a deploy, a canary window and a rollback budget on noise. 0.01 is one
# standard deviation above zero; asking for more would mean real gains never ship, since
# the whole spread between the best and worst configuration in the Lab 2 study was 0.028.
MIN_IMPROVEMENT = 0.01


def incumbent_from_registry(metric_name: str, alias: str) -> float | None:
    """Read the production model's own recorded score from the registry.

    The metric is read from the *version's* lineage tag, not recomputed. Recomputing it
    here would silently compare a candidate scored on today's split against an incumbent
    rescored on today's data, which is a different comparison from the one this gate
    claims to make.

    Returns None when nothing is registered under the alias — a genuinely first run.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from src import config

    loaded_config = config.load(strict=False)
    mlflow.set_tracking_uri(loaded_config.mlflow_tracking_uri)
    client = MlflowClient()

    try:
        version = client.get_model_version_by_alias(loaded_config.model_registry_name, alias)
    except Exception as error:
        print(f"no '{alias}' alias in the registry ({type(error).__name__})")
        return None

    tag_name = "metric_test" if metric_name.startswith("test") else "metric_val"
    recorded = version.tags.get(tag_name)
    if recorded is None:
        print(f"version {version.version} carries no {tag_name} tag — lineage is incomplete")
        return None

    print(f"incumbent is version {version.version} (alias '{alias}')")
    return float(recorded)


def main() -> int:
    """Decide whether the candidate model may be registered."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--metrics", type=Path, required=True)
    argument_parser.add_argument("--metric", default="test_roc_auc")
    argument_parser.add_argument("--incumbent", type=float, default=None,
                    help="current production score; omit to read from the registry")
    argument_parser.add_argument("--alias", default="production",
                    help="registry alias holding the incumbent")
    argument_parser.add_argument("--decision-out", type=Path, default=None,
                    help="write 'pass' or 'fail' to this path. The pipeline's condition "
                         "reads it, which is what makes the gate a branch in the DAG "
                         "rather than a message in a log nobody reads.")
    argument_parser.add_argument("--allow-first-registration", action="store_true",
                    help="pass when nothing is registered yet. Off by default so that a "
                         "registry lookup that quietly fails cannot be mistaken for an "
                         "empty registry and wave a bad model through.")
    options = argument_parser.parse_args()

    payload = json.loads(options.metrics.read_text())
    candidate = float(payload[options.metric])
    print(f"candidate {options.metric} = {candidate:.4f}")

    def finish(decision: str, exit_code: int) -> int:
        """Record the decision for the pipeline condition, then return the exit code."""
        if options.decision_out:
            options.decision_out.parent.mkdir(parents=True, exist_ok=True)
            options.decision_out.write_text(decision)
        return exit_code

    if candidate < MIN_ABSOLUTE:
        print(f"GATE FAIL  below absolute floor {MIN_ABSOLUTE} "
              f"(a single-feature heuristic scores 0.8257 on this data)")
        return finish("fail", 1)

    incumbent = options.incumbent
    if incumbent is None:
        incumbent = incumbent_from_registry(options.metric, options.alias)

    if incumbent is None:
        if options.allow_first_registration:
            print("no incumbent — first registration allowed by flag")
            print("GATE PASS")
            return finish("pass", 0)
        print("GATE FAIL  no incumbent found and --allow-first-registration was not given")
        return finish("fail", 1)

    delta = candidate - incumbent
    print(f"incumbent {incumbent:.4f}  delta {delta:+.4f}  required {MIN_IMPROVEMENT:+.4f}")
    if delta < MIN_IMPROVEMENT:
        print("GATE FAIL  improvement within noise; not registering")
        return finish("fail", 1)

    print("GATE PASS")
    return finish("pass", 0)


if __name__ == "__main__":
    raise SystemExit(main())
