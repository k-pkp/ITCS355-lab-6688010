"""Lab 4 — deliberately shift a feature's distribution.

    python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 6

Three modes, and they are not equivalent:
  shift   moves the mean         — KS reacts strongly, PSI moderately
  scale   changes the spread     — PSI reacts, the mean barely moves
  mix     reweights the machines — the realistic one, and the hardest to spot

Run your detector against each. Which statistic catches which is exactly what Quiz 4 asks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def main() -> int:
    """Shift one feature's distribution on purpose and write the drifted window."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--source", type=Path, default=Path("data/raw/sensors.csv"))
    argument_parser.add_argument("--out", type=Path, default=Path("data/current.csv"))
    argument_parser.add_argument("--feature", default="temp_c")
    argument_parser.add_argument("--mode", choices=["shift", "scale", "mix"], default="shift")
    argument_parser.add_argument("--magnitude", type=float, default=6.0)
    argument_parser.add_argument("--seed", type=int, default=7)
    options = argument_parser.parse_args()

    rng = np.random.default_rng(options.seed)
    frame = pd.read_csv(options.source)
    before = frame[options.feature].mean()

    if options.mode == "shift":
        frame[options.feature] = frame[options.feature] + options.magnitude
    elif options.mode == "scale":
        mean = frame[options.feature].mean()
        frame[options.feature] = mean + (frame[options.feature] - mean) * options.magnitude
    else:  # mix — over-sample a subset of machines, as a fleet change would
        machines = frame["machine_id"].unique()
        favoured = rng.choice(machines, size=max(1, len(machines) // 5), replace=False)
        weights = np.where(frame["machine_id"].isin(favoured), 5.0, 1.0)
        frame = frame.sample(n=len(frame), replace=True, weights=weights, random_state=options.seed)

    options.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(options.out, index=False, lineterminator="\n")
    print(f"wrote {options.out}  mode={options.mode}  feature={options.feature}")
    print(f"  mean {before:.4f} -> {frame[options.feature].mean():.4f}")
    print(f"\nNow run: python -m monitoring.drift --current {options.out}")
    print("Record the time from injection to alert. That is your detection latency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
