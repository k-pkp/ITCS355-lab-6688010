"""Lab 2 — rank tracked runs by metric AND by cost per point.

    python scripts/compare_runs.py --experiment itcs355-lab2

Writes reports/lab2-comparison.md. The cost-per-point column is what the lab is about:
the highest-scoring run is frequently not the one you should register.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd

from src import config


# Written for Lab 2 Task 3, and regenerated with the table so the two can never disagree.
JUSTIFICATION = """
Registered: 200 trees, max_depth 6, min_samples_leaf 5, max_features sqrt, no class
weighting — trial `b6dc2ba6`, val_roc_auc 0.8416. The best trial scored 0.8463.

**Why not the highest scorer.** The 0.0047 gap is smaller than the noise. Across five
seeds the top configuration averages val 0.8577 (sd 0.0127) and the registered one 0.8556
(sd 0.0138); the intervals overlap almost completely, so the ranking between them is a
coin toss re-flipped by the split. What is not noise is the price: 500 trees cost 0.0015
THB per fit against 0.0006, 2.5x for a difference we cannot measure, and that multiple
follows the model into every retrain and every prediction.

**Seed variance.** val 0.8556 ± 0.0138, test 0.8540 ± 0.0087 over seeds
20260101–20260105. Any claim finer than the second decimal place is unsupported.

**Cost.** 0.0006 THB per fit at the verified e2-standard-4 rate (5.443 THB/h,
asia-southeast1). Weekly retraining is about 0.03 THB a month of compute; the pipeline
around it, not the fit, is what costs money.

**How this could be wrong.** Sixty machines per split is thin. If the real cohort is
larger and more varied, depth 6 will underfit where depth 10 would not, and the seed
noise that hides the difference here would shrink until it is real.
"""


def main() -> int:
    """Rank the tracked trials by metric and by cost per point, and write the report."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--experiment", default="itcs355-lab2")
    argument_parser.add_argument("--metric", default="val_roc_auc")
    argument_parser.add_argument("--out", type=Path, default=Path("reports/lab2-comparison.md"))
    options = argument_parser.parse_args()

    loaded_config = config.load(strict=False)
    mlflow.set_tracking_uri(loaded_config.mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name(options.experiment)
    if exp is None:
        print(f"No experiment named {options.experiment!r}. Run `make tune` first.")
        return 1

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        print("No runs found.")
        return 1

    metric_col = f"metrics.{options.metric}"
    cost_col = "metrics.cost_thb"

    # Only priced runs belong in a cost comparison. The experiment also holds the
    # registration run, which reuses a configuration already measured here; leaving it in
    # would double-count one configuration and put a blank in the cost column.
    runs = runs[runs[cost_col].notna()]
    if runs.empty:
        print("No priced trials found. Run `make tune` first.")
        return 1

    baseline = runs[metric_col].min()

    table = pd.DataFrame({
        "run_id": runs["run_id"].str[:8],
        options.metric: runs[metric_col].round(4),
        "cost_thb": runs.get(cost_col, 0).round(4),
        "n_estimators": runs.get("params.n_estimators"),
        "max_depth": runs.get("params.max_depth"),
        "min_samples_leaf": runs.get("params.min_samples_leaf"),
        "max_features": runs.get("params.max_features"),
        "class_weight": runs.get("params.class_weight"),
    })
    gain = (table[options.metric] - baseline).clip(lower=1e-9)
    table["thb_per_point"] = (table["cost_thb"] / (gain * 100)).round(4)
    table = table.sort_values(options.metric, ascending=False)

    options.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lab 2 — Run comparison",
        "",
        f"Experiment `{options.experiment}` · {len(table)} trials · "
        f"total spend {table['cost_thb'].sum():.4f} THB",
        "",
        "`thb_per_point` is cost per percentage point of "
        f"{options.metric} above the worst trial. Cheap improvements rank low; expensive "
        "improvements rank high, however good the headline number is.",
        "",
        table.to_markdown(index=False),
        "",
        "## Which model did you register, and why?",
        "",
        JUSTIFICATION.strip(),
    ]
    options.out.write_text("\n".join(lines))
    print(f"wrote {options.out}  ({len(table)} trials)")
    print(table.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
