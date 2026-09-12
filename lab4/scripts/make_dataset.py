"""Generate the default dataset for Lab 1.

Deterministic: the same seed always produces a byte-identical file, so the data
fingerprint is stable across machines. Swap this out if you bring your own problem —
the rest of the repo does not care where data/raw/sensors.csv came from.

Machines have persistent characteristics, which is exactly why the split must be
grouped by machine_id. See src/data.split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_MACHINES = 240
READINGS_PER_MACHINE = 25


def build(seed: int) -> pd.DataFrame:
    """Generate the synthetic sensor readings for one seed, deterministically."""
    rng = np.random.default_rng(seed)
    rows = []
    reading_id = 0
    for machine in range(N_MACHINES):
        # Persistent per-machine traits. A row-wise split would let a model memorise these.
        base_temp = rng.normal(72, 9)
        wear = rng.gamma(2.0, 1.4)
        duty = rng.uniform(0.3, 1.0)
        for _ in range(READINGS_PER_MACHINE):
            hours = float(rng.uniform(0, 9000))
            temperature = base_temp + 0.0016 * hours + rng.normal(0, 2.2)
            vib = 1.4 + 0.55 * wear + 0.00035 * hours + rng.normal(0, 0.45)
            pressure = 320 - 0.7 * wear + rng.normal(0, 14)
            load = float(np.clip(100 * duty + rng.normal(0, 6), 0, 100))
            humidity = float(np.clip(rng.normal(58, 12), 0, 100))

            logit = (
                -6.1
                + 0.052 * (temperature - 72)
                + 0.71 * (vib - 2.2)
                + 0.00021 * hours
                + 0.019 * (load - 60)
                - 0.004 * (pressure - 320)
            )
            probability = 1.0 / (1.0 + np.exp(-logit))
            rows.append({
                "reading_id": reading_id,
                "machine_id": machine,
                "temp_c": round(float(temperature), 3),
                "vibration_mm_s": round(float(vib), 3),
                "pressure_kpa": round(float(pressure), 3),
                "hours_since_service": round(hours, 3),
                "load_pct": round(load, 3),
                "ambient_humidity": round(humidity, 3),
                "failed_within_7d": int(rng.random() < probability),
            })
            reading_id += 1
    return pd.DataFrame(rows)


def main() -> None:
    """Generate the dataset and write it where the pipeline expects to find it."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--seed", type=int, default=20260101)
    argument_parser.add_argument("--out", type=Path, default=Path("data/raw/sensors.csv"))
    options = argument_parser.parse_args()

    frame = build(options.seed)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    # DELIBERATELY BROKEN, for Lab 4 Task 3. A plausible-looking "cleanup": ambient
    # humidity reads noisy and carries only 4.9% of the model's importance, so someone
    # drops it from the export to save a column. The generator still runs, the file still
    # loads, and nothing complains until a test asserts the schema.
    frame = frame.drop(columns=["ambient_humidity"])
    frame.to_csv(options.out, index=False, lineterminator="\n")
    rate = frame["failed_within_7d"].mean()
    print(f"wrote {options.out}  rows={len(frame)}  machines={frame.machine_id.nunique()}  positive_rate={rate:.3f}")


if __name__ == "__main__":
    main()
