"""Lab 3 Task 4 — run the canary, detect the degradation from metrics, roll back.

    python loadtest/canary_experiment.py --requests 20000

Assumes three processes are already up: the incumbent on 8081, the canary on 8082, and
loadtest/canary_proxy.py on 8080 splitting between them.

The detector below is deliberately blind. It sees two slots called blue and green, their
response times, their predicted probabilities, and — seven simulated days later — the
outcomes. It is never told which slot holds the new model, because in production nothing
tells you that either; the version is a fact about your deployment, not about the traffic.

Two signals are tracked, because they become available at very different times:

  * prediction distribution, available immediately. A model whose probabilities are
    calibrated differently shifts the mean of what it emits within a few hundred requests.
  * ranking quality, available only once outcomes arrive. Truer, and a week late.

Which of the two catches the problem first is the finding the report turns on.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import roc_auc_score

from src import config, data, seeds

PROXY = "http://127.0.0.1:8080"


def replay_pool() -> list[tuple[dict, int]]:
    """Held-out rows and their outcomes, used as a stand-in for production traffic.

    Rows are drawn from this pool with replacement. Repeating a row does not add
    information about the model, so the request counts below are an optimistic view of how
    quickly real traffic would decide the question; the report says so.
    """
    loaded_config = config.load(strict=False)
    frame = data.load_raw(loaded_config.raw_path)
    seeds.set_all(20260101)
    _, validation_frame, test_frame = data.split(frame, seed=20260101)

    pool: list[tuple[dict, int]] = []
    for source_frame in (validation_frame, test_frame):
        for _, row in source_frame.iterrows():
            features = {name: float(row[name]) for name in data.FEATURES}
            pool.append((features, int(row[data.TARGET])))
    return pool


def send_one(features: dict) -> tuple[str, float]:
    """Send one prediction through the proxy. Returns the serving slot and probability."""
    request = urllib.request.Request(
        f"{PROXY}/predict",
        data=json.dumps(features).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(request) as response:
        slot = response.headers.get("x-canary-slot", "unknown")
        body = json.load(response)
    return slot, float(body["probability"])


def set_weights(weights: dict[str, int]) -> dict:
    """Move traffic. This is the rollback action, and it is timestamped by the proxy."""
    request = urllib.request.Request(
        f"{PROXY}/admin/weights",
        data=json.dumps(weights).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


class SlotMetrics:
    """Rolling window of what one slot emitted, and how it turned out."""

    def __init__(self, window: int) -> None:
        """Start an empty window of the given size."""
        self.window = window
        self.probabilities: list[float] = []
        self.outcomes: list[int] = []

    def record(self, probability: float, outcome: int) -> None:
        """Add one observation, discarding anything older than the window."""
        self.probabilities.append(probability)
        self.outcomes.append(outcome)
        if len(self.probabilities) > self.window:
            self.probabilities.pop(0)
            self.outcomes.pop(0)

    def mean_probability(self) -> float | None:
        """Mean predicted probability, available with no outcomes at all."""
        if len(self.probabilities) < 100:
            return None
        return statistics.mean(self.probabilities)

    def ranking_quality(self) -> float | None:
        """Windowed ROC AUC, available only once outcomes are known."""
        if len(self.outcomes) < 400 or len(set(self.outcomes)) < 2:
            return None
        return float(roc_auc_score(self.outcomes, self.probabilities))


def parse_command_line() -> argparse.Namespace:
    """Command line for the canary experiment."""
    parser = argparse.ArgumentParser(description="Canary, detection, and rollback")
    parser.add_argument("--requests", type=int, default=20000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--window", type=int, default=1500)
    parser.add_argument("--distribution-margin", type=float, default=0.05,
                        help="mean predicted probability gap that counts as divergence")
    parser.add_argument("--quality-margin", type=float, default=0.02,
                        help="ROC AUC gap that counts as a real degradation")
    parser.add_argument("--confirmations", type=int, default=3,
                        help="consecutive checks a signal must hold before it fires")
    parser.add_argument("--check-every", type=int, default=250)
    parser.add_argument("--after-rollback", type=int, default=1000,
                        help="requests to send after rollback, as evidence traffic moved")
    parser.add_argument("--out", type=Path, default=Path("reports/lab3-canary.md"))
    parser.add_argument("--seed", type=int, default=20260101)
    return parser.parse_args()


def main() -> int:
    """Drive traffic, watch two signals, roll back when one of them fires."""
    options = parse_command_line()
    random.seed(options.seed)
    pool = replay_pool()

    set_weights({"blue": 90, "green": 10})
    slots = {"blue": SlotMetrics(options.window), "green": SlotMetrics(options.window)}
    served = {"blue": 0, "green": 0}

    distribution_streak = 0
    quality_streak = 0
    distribution_alert: dict | None = None
    quality_alert: dict | None = None
    timeline: list[str] = []

    started_at = time.time()
    print(f"canary running at 90/10, {options.requests} requests\n")

    with ThreadPoolExecutor(max_workers=options.concurrency) as pool_executor:
        sent = 0
        while sent < options.requests and quality_alert is None:
            batch_size = min(options.check_every, options.requests - sent)
            draws = [random.choice(pool) for _ in range(batch_size)]
            results = list(pool_executor.map(lambda draw: send_one(draw[0]), draws))

            for (features, outcome), (slot, probability) in zip(draws, results):
                slots[slot].record(probability, outcome)
                served[slot] += 1
            sent += batch_size

            blue_mean = slots["blue"].mean_probability()
            green_mean = slots["green"].mean_probability()
            if blue_mean is not None and green_mean is not None:
                gap = abs(blue_mean - green_mean)
                distribution_streak = distribution_streak + 1 if gap > options.distribution_margin else 0
                if distribution_streak >= options.confirmations and distribution_alert is None:
                    distribution_alert = {
                        "requests": sent, "seconds": time.time() - started_at,
                        "blue": blue_mean, "green": green_mean, "gap": gap,
                    }
                    timeline.append(
                        f"{sent:>6} requests · {time.time() - started_at:6.1f}s · "
                        f"DISTRIBUTION ALERT — mean predicted probability blue {blue_mean:.3f} "
                        f"vs green {green_mean:.3f} (gap {gap:.3f})")
                    print(timeline[-1])

            blue_quality = slots["blue"].ranking_quality()
            green_quality = slots["green"].ranking_quality()
            if blue_quality is not None and green_quality is not None:
                quality_gap = blue_quality - green_quality
                quality_streak = quality_streak + 1 if abs(quality_gap) > options.quality_margin else 0
                if quality_streak >= options.confirmations:
                    worse_slot = "green" if quality_gap > 0 else "blue"
                    quality_alert = {
                        "requests": sent, "seconds": time.time() - started_at,
                        "blue": blue_quality, "green": green_quality,
                        "worse_slot": worse_slot,
                    }
                    timeline.append(
                        f"{sent:>6} requests · {time.time() - started_at:6.1f}s · "
                        f"QUALITY ALERT — windowed AUC blue {blue_quality:.4f} vs green "
                        f"{green_quality:.4f}; {worse_slot} is worse")
                    print(timeline[-1])

        if quality_alert is None:
            print("no quality alert fired within the request budget")
            worse_slot = "green"
        else:
            worse_slot = quality_alert["worse_slot"]

        healthy_slot = "blue" if worse_slot == "green" else "green"
        rollback = set_weights({healthy_slot: 100, worse_slot: 0})
        timeline.append(f"       rollback · {rollback['iso']} · weights {rollback['from']} -> {rollback['to']}")
        print(timeline[-1])

        after = {"blue": 0, "green": 0}
        draws = [random.choice(pool) for _ in range(options.after_rollback)]
        for slot, _ in pool_executor.map(lambda draw: send_one(draw[0]), draws):
            after[slot] += 1

    timeline.append(
        f"       post-rollback traffic · blue {after['blue']} · green {after['green']} "
        f"(of {options.after_rollback})")
    print(timeline[-1])

    write_report(options, served, after, slots, distribution_alert, quality_alert, timeline)
    print(f"\nwrote {options.out}")
    return 0


def write_report(options, served, after, slots, distribution_alert, quality_alert, timeline) -> None:
    """Write the canary section of the Lab 3 report."""
    def describe(alert: dict | None, label: str) -> str:
        """Render one signal's outcome as a report bullet."""
        if alert is None:
            return f"- **{label}**: never fired within the request budget."
        return (f"- **{label}**: fired after {alert['requests']} requests "
                f"({alert['seconds']:.1f}s of wall clock).")

    lines = [
        "# Lab 3 Task 4 — canary, detection, rollback",
        "",
        "Incumbent (registry version 1) in slot **blue**, canary (registry version 3) in slot",
        "**green**, split 90/10 by `loadtest/canary_proxy.py`. The detector is told the slot",
        "names and nothing else.",
        "",
        "## Timeline",
        "",
        "```",
        *timeline,
        "```",
        "",
        "## Detection",
        "",
        describe(distribution_alert, "Prediction distribution (no outcomes needed)"),
        describe(quality_alert, "Windowed ROC AUC (needs outcomes)"),
        "",
        f"Traffic served during the canary: blue {served['blue']}, green {served['green']}.",
        f"Traffic served after rollback: blue {after['blue']}, green {after['green']}.",
        "",
        "The decision log at `reports/canary-decisions.jsonl` carries a timestamped record of",
        "every routed request and of the weight change itself.",
        "",
        "## Caveat on the request counts",
        "",
        "Traffic is replayed from held-out rows drawn with replacement. Repeating a row adds",
        "no new information about the model, so the counts above are the optimistic end of",
        "how quickly live traffic would settle the question.",
    ]
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
