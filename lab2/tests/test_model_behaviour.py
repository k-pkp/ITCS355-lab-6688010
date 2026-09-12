"""Lab 4 — model behaviour tests.

These assert what the model DOES, independent of its metric. A model can score 0.85 and
still be broken in ways these catch, and unlike a metric threshold they do not drift out
of relevance as the data changes.

Quiz 4 asks you to distinguish these from data contract tests. The difference: a contract
test fails when an upstream producer changes something; a behaviour test fails when the
model changes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import config, data, seeds

RAW = config.REPO_ROOT / "data" / "raw" / "sensors.csv"
# Derived from the p95 target in loadtest/k6.js, which is 100 ms end to end for one
# /predict. Measured breakdown at that target: the HTTP layer, validation and DataFrame
# construction cost about 2 ms, and the rest of the budget belongs to the network and to
# queueing under load. Giving the model 15 ms of it leaves the service a real margin, and
# is still twice what the registered forest actually needs (8.0 ms at n_jobs=1).
LATENCY_BUDGET_MS = 15.0


@pytest.fixture(scope="module")
def fitted():
    """A model trained once for the module, with the test split it has not seen."""
    if not RAW.exists():
        pytest.skip("run `make data` first")
    seed = seeds.set_all()
    frame = data.load_raw(RAW)
    train_df, _, test_df = data.split(frame, seed=seed)
    # n_jobs=1 because this test asks a serving question, and serving scores one row at a
    # time. With n_jobs=-1 joblib's fan-out costs more than the forest does — 37 ms against
    # 8 ms — which is the bug this suite exists to catch, so measuring with it on would
    # measure the wrong thing. See service.app.tune_for_single_row_serving.
    model = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=5,
                                   random_state=seed, n_jobs=1)
    model.fit(train_df[data.FEATURES], train_df[data.TARGET])
    return model, test_df


def test_predictions_are_valid_probabilities(fitted):


    """Every prediction is a real number in [0, 1], with no NaN."""
    model, test_df = fitted
    proba = model.predict_proba(test_df[data.FEATURES])[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
    assert not np.isnan(proba).any()


def test_known_healthy_machine_scores_low(fitted):
    """A cool, new, lightly loaded machine must not be flagged as about to fail.

    This is a domain assertion. If it fails, the model has learned something backwards,
    and no aggregate metric would have told you.
    """
    model, _ = fitted
    healthy = pd.DataFrame([{
        "temp_c": 66.0, "vibration_mm_s": 1.4, "pressure_kpa": 330.0,
        "hours_since_service": 50.0, "load_pct": 30.0, "ambient_humidity": 50.0,
    }])[data.FEATURES]
    assert model.predict_proba(healthy)[0, 1] < 0.5


def test_risk_increases_with_wear(fitted):
    """Monotonicity the domain requires: more hours since service, higher risk.

    Tree ensembles are not monotonic by construction, so this can genuinely fail. If it
    does, that is information — not a reason to delete the test.
    """
    model, _ = fitted
    base = {
        "temp_c": 82.0, "vibration_mm_s": 4.0, "pressure_kpa": 310.0,
        "load_pct": 75.0, "ambient_humidity": 60.0,
    }
    rows = pd.DataFrame([{**base, "hours_since_service": hours_since_service} for hours_since_service in (100, 3000, 8500)])[data.FEATURES]
    proba = model.predict_proba(rows)[:, 1]
    assert proba[0] <= proba[-1], f"risk fell as wear rose: {proba}"


def test_prediction_latency_within_budget(fitted):


    """One row scores inside the share of the latency budget allotted to the model."""
    model, test_df = fitted
    sample = test_df[data.FEATURES].head(100)
    model.predict_proba(sample)  # warm up; the first call includes lazy setup
    started = time.perf_counter()
    for _ in range(20):
        model.predict_proba(sample.head(1))
    elapsed_ms = (time.perf_counter() - started) * 1000 / 20
    assert elapsed_ms < LATENCY_BUDGET_MS, f"single prediction took {elapsed_ms:.2f} ms"


def test_model_is_not_constant(fitted):
    """A model that always predicts the majority class can post a respectable accuracy
    and is useless. Catch it here rather than in production."""
    model, test_df = fitted
    proba = model.predict_proba(test_df[data.FEATURES])[:, 1]
    assert proba.std() > 0.01, "predictions have almost no variance"
