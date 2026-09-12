"""ITCS355 Lab 3 — load test (Locust alternative to k6).

    locust -f loadtest/locustfile.py --host https://<endpoint> \
           --users 10 --spawn-rate 5 --run-time 60s --headless

Use either this or k6, not both. Whichever you choose, commit it.
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task


def sample_payload() -> dict:
    """One randomised sensor reading inside the schema's declared bounds."""
    return {
        "temp_c": round(random.uniform(60, 95), 3),
        "vibration_mm_s": round(random.uniform(1.0, 8.0), 3),
        "pressure_kpa": round(random.uniform(280, 350), 3),
        "hours_since_service": round(random.uniform(0, 9000), 3),
        "load_pct": round(random.uniform(20, 100), 3),
        "ambient_humidity": round(random.uniform(30, 85), 3),
    }


class PredictUser(HttpUser):
    wait_time = between(0.0, 0.1)

    @task(9)
    def predict(self):
        """Single-prediction path: the shape the stated latency target is measured against."""
        with self.client.post("/predict", json=sample_payload(), catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")

    @task(1)
    def predict_batch(self):
        """Batch path, measured at 90.5x cheaper per row than the same rows sent singly.

                reports/lab3-load.md carries the comparison and the point at which the
                advantage stops growing.
                """
        rows = [sample_payload() for _ in range(50)]
        # TODO(Lab 3): compare this against 50 single calls. Report the difference,
        # and the concurrency at which the advantage disappears.
        self.client.post("/predict/batch", json={"rows": rows})
