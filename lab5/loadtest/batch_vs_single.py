"""Lab 3 Task 3 — measure batching and payload size against the running service.

    python loadtest/batch_vs_single.py --target http://127.0.0.1:8080

Two questions the load test alone does not answer:

  1. Does one /predict/batch call with 100 rows beat 100 calls to /predict, and by how much?
  2. Where does payload size stop being free and start dominating?

Both are measured against a service that is already warm. Numbers land in
reports/lab3-load.md.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request


def sample_row() -> dict[str, float]:
    """One plausible sensor reading, inside the schema's declared bounds."""
    return {
        "temp_c": round(random.uniform(60, 95), 3),
        "vibration_mm_s": round(random.uniform(1.0, 8.0), 3),
        "pressure_kpa": round(random.uniform(280, 350), 3),
        "hours_since_service": round(random.uniform(0, 9000), 3),
        "load_pct": round(random.uniform(20, 100), 3),
        "ambient_humidity": round(random.uniform(30, 85), 3),
    }


def post_json(url: str, body: dict) -> tuple[dict, float]:
    """POST one JSON body and return the parsed response and the wall-clock milliseconds.

    A fresh connection per call, deliberately: reusing a keep-alive connection against a
    multi-worker uvicorn adds a ~40 ms delayed-ACK stall to every request, which would
    swamp the effect being measured here. That artefact is documented in the report.
    """
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        parsed = json.load(response)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return parsed, elapsed_ms


def measure_single_calls(target: str, row_count: int) -> tuple[float, list[float]]:
    """Score `row_count` rows one request at a time. Returns total ms and per-call ms."""
    per_call: list[float] = []
    started = time.perf_counter()
    for _ in range(row_count):
        _, elapsed_ms = post_json(f"{target}/predict", sample_row())
        per_call.append(elapsed_ms)
    total_ms = (time.perf_counter() - started) * 1000
    return total_ms, per_call


def measure_batch_call(target: str, row_count: int) -> float:
    """Score `row_count` rows in one batch request. Returns total ms."""
    rows = [sample_row() for _ in range(row_count)]
    _, elapsed_ms = post_json(f"{target}/predict/batch", {"rows": rows})
    return elapsed_ms


def parse_command_line() -> argparse.Namespace:
    """Command line for the batch and payload measurements."""
    parser = argparse.ArgumentParser(description="Batch and payload-size measurements")
    parser.add_argument("--target", default="http://127.0.0.1:8080")
    parser.add_argument("--rows", type=int, default=100, help="rows in the batch comparison")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260101)
    return parser.parse_args()


def main() -> int:
    """Run both measurements and print a table for the report."""
    options = parse_command_line()
    random.seed(options.seed)

    # Warm the path so the first call's import cost does not land in the numbers.
    post_json(f"{options.target}/predict", sample_row())

    print(f"Batching: {options.rows} rows, {options.repeats} repeats\n")
    single_totals: list[float] = []
    batch_totals: list[float] = []
    for _ in range(options.repeats):
        total_ms, _ = measure_single_calls(options.target, options.rows)
        single_totals.append(total_ms)
        batch_totals.append(measure_batch_call(options.target, options.rows))

    single_median = statistics.median(single_totals)
    batch_median = statistics.median(batch_totals)
    print(f"  {options.rows} separate /predict calls   {single_median:9.2f} ms total"
          f"  ({single_median / options.rows:6.3f} ms/row)")
    print(f"  one /predict/batch call        {batch_median:9.2f} ms total"
          f"  ({batch_median / options.rows:6.3f} ms/row)")
    print(f"  batching is {single_median / batch_median:.1f}x faster for the same rows\n")

    print("Payload size: one batch request, varying row count\n")
    print(f"  {'rows':>6} {'total ms':>10} {'ms/row':>9} {'bytes':>9}")
    for row_count in (1, 10, 25, 50, 100):
        samples = []
        for _ in range(options.repeats):
            samples.append(measure_batch_call(options.target, row_count))
        median_ms = statistics.median(samples)
        body_bytes = len(json.dumps({"rows": [sample_row() for _ in range(row_count)]}))
        print(f"  {row_count:>6} {median_ms:>10.2f} {median_ms / row_count:>9.3f} {body_bytes:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
