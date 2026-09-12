"""Lab 4 — drift detection.

PSI and Kolmogorov–Smirnov, implemented directly so you can see what the numbers mean.
Evidently is allowed instead, but you must still be able to explain what your chosen
statistic measures and why your threshold is what it is.

    python -m monitoring.drift --reference data/raw/sensors.csv --current data/current.csv

A threshold copied from a tutorial is not a justified threshold, and Lab 4 grades the
justification, not the code.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Conventional PSI reading, and it IS only conventional — it comes from credit scoring,
# where features are stable and volumes are large.
PSI_NO_CHANGE = 0.10
PSI_MODERATE = 0.25

# OUR alerting threshold, and the reason for it. Both numbers below were measured on this
# dataset on 2026-09-07; the working is in reports/lab4-drift.md.
#
#   Noise floor. PSI between the reference and a 600-row window drawn from the SAME
#   unchanged data peaked at 0.043 over 200 draws, across all six features. Anything at or
#   below that is the window size talking, not the fleet.
#
#   Harm floor. Shifting the most important feature (vibration_mm_s) by one standard
#   deviation reaches PSI 1.03 and costs 0.0007 ROC AUC — nothing. It takes a two-sigma
#   shift, PSI 3.54, before the metric moves by 0.011. Ranking survives a uniform shift
#   because the ordering within the cohort barely changes.
#
# So there is a very wide band between "measurably not noise" and "measurably harmful",
# and the threshold is a choice about which of those two things you want to be told.
# 0.20 sits near the bottom of that band: 4.6x the noise ceiling, so it will not cry wolf,
# and far below the level at which the model actually suffers.
#
# It is set low on purpose. The label for this problem — failed_within_7d — arrives seven
# days late, so the input-side statistic is the only signal available on the day something
# changes. Its job is to start an investigation, not to trigger a retrain: a PSI breach
# here is a question ("did an upstream producer change?"), and the answer decides whether
# anything should happen at all.
PSI_ALERT = 0.20


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ref_mean: float
    cur_mean: float
    verdict: str


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index.

    Bins the reference into quantiles and compares the proportion of mass falling in each.
    Sensitive to changes in shape, not only in mean — which is why a feature can drift
    badly while its average looks untouched.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing: an empty bin would otherwise make the log term infinite.
    ref_prop = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_prop = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: the largest gap between the two CDFs.

    Complements PSI. KS is more sensitive to a shift in location; PSI to a change in shape.
    Reporting both, and noticing when they disagree, is worth more than either alone.
    """
    ref = np.sort(reference)
    cur = np.sort(current)
    pooled = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, pooled, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, pooled, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def verdict_for(score: float) -> str:
    """Label a PSI score using the conventional credit-scoring bands.

    Kept separate from PSI_ALERT deliberately: the bands are what the literature says, the
    alert threshold is what this system decided, and conflating them is how a tutorial
    default ends up in production wearing the costume of a decision.
    """
    if score < PSI_NO_CHANGE:
        return "stable"
    if score < PSI_MODERATE:
        return "moderate"
    return "significant"


def compare(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[FeatureDrift]:
    """Score every feature for drift, worst first."""
    results = []
    for feature in features:
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        score = psi(ref, cur)
        results.append(FeatureDrift(
            feature=feature,
            psi=round(score, 5),
            ks_statistic=round(ks_statistic(ref, cur), 5),
            ref_mean=round(float(ref.mean()), 4),
            cur_mean=round(float(cur.mean()), 4),
            verdict=verdict_for(score),
        ))
    return sorted(results, key=lambda feature_result: feature_result.psi, reverse=True)


def main() -> int:
    """Compare a current window against the reference and alert above the threshold."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, data

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--reference", type=Path, default=Path("data/raw/sensors.csv"))
    argument_parser.add_argument("--current", type=Path, required=True)
    argument_parser.add_argument("--threshold", type=float, default=PSI_ALERT,
                    help="alert above this PSI. The default is justified where PSI_ALERT is defined.")
    argument_parser.add_argument("--out", type=Path, default=Path("reports/drift.json"))
    argument_parser.add_argument("--emit", action="store_true", help="send scores as cloud metrics")
    options = argument_parser.parse_args()

    reference = pd.read_csv(options.reference)
    current = pd.read_csv(options.current)
    results = compare(reference, current, data.FEATURES)

    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps([asdict(feature_result) for feature_result in results], indent=2))

    print(f"{'feature':<22}{'psi':>10}{'ks':>10}  verdict")
    for feature_result in results:
        print(f"{feature_result.feature:<22}{feature_result.psi:>10.5f}{feature_result.ks_statistic:>10.5f}  {feature_result.verdict}")

    if options.emit:
        # Reaches Cloud Monitoring through the adapter, which is what puts the drift panel
        # on the dashboard. The metric name is provider-neutral here; the adapter turns it
        # into whatever the provider's time-series API wants.
        from cloudlayer.factory import get_adapter
        adapter = get_adapter(config.load(strict=False))
        for feature_result in results:
            adapter.emit_metric(f"drift.psi.{feature_result.feature}", feature_result.psi)

    breached = [feature_result for feature_result in results if feature_result.psi >= options.threshold]
    if breached:
        print(f"\nALERT  {len(breached)} feature(s) above threshold {options.threshold}: "
              + ", ".join(feature_result.feature for feature_result in breached))
        print("Before you retrain: is this drift, or is it a broken upstream pipeline? "
              "Retraining on corrupted data destroys a working model faster than any "
              "schedule would.")
        return 2
    print(f"\nOK  no feature above threshold {options.threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
