"""Contract tests for the LLM cost model and the evaluation gate. Lab 5 Part B.

The gate is the thing that has to be trustworthy. A gate that cannot fail is decoration,
so most of what follows checks that it fails when it should — the same reasoning as
tests/test_data.py, applied to text.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import llmcost  # noqa: E402

GOLDEN = ROOT / "evals" / "golden" / "triage.jsonl"
BASELINE = ROOT / "evals" / "fixtures" / "triage-baseline.jsonl"
REGRESSED = ROOT / "evals" / "fixtures" / "triage-regressed.jsonl"


def run_eval(*extra_arguments: str) -> subprocess.CompletedProcess:
    """Run the eval harness as a subprocess, so its exit code is part of what is tested.

    Importing and calling main() would test the scoring but not the gate, because the
    gate's whole contract is the non-zero exit that stops a pipeline.
    """
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "llm_eval.py"), *extra_arguments],
        capture_output=True, text=True, cwd=ROOT,
    )


# --- the golden set is a data contract of its own ----------------------------

def test_golden_set_is_wellformed():

    """Every golden case has an id, a prompt and at least one check that can fail."""
    ids = []
    for line in GOLDEN.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        assert case["id"] and case["prompt"], "every case needs an id and a prompt"
        assert case["checks"], f"{case['id']} asserts nothing, so it can never fail"
        ids.append(case["id"])
    assert len(ids) == len(set(ids)), "duplicate case ids"


def test_every_golden_case_has_a_recorded_response():


    """No case is silently unscored because its recorded response is missing."""
    cases = {json.loads(case)["id"] for case in GOLDEN.read_text().splitlines() if case.strip()}
    for fixture in (BASELINE, REGRESSED):
        got = {json.loads(case)["id"] for case in fixture.read_text().splitlines() if case.strip()}
        assert cases <= got, f"{fixture.name} is missing {sorted(cases - got)}"


# --- the gate ----------------------------------------------------------------

def test_baseline_fixture_passes_every_case(tmp_path):

    """The baseline is a clean starting point, so any later failure is a real regression."""
    out = tmp_path / "report.json"
    proc = run_eval("--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    assert report["pass_rate"] == 1.0, [result for result in report["results"] if not result["passed"]]


def test_gate_fails_on_a_regressed_response(tmp_path):


    """The gate rejects a set where a previously passing case now fails."""
    base = tmp_path / "base.json"
    assert run_eval("--out", str(base)).returncode == 0

    proc = run_eval("--responses", str(REGRESSED),
                    "--out", str(tmp_path / "now.json"), "--baseline", str(base))
    assert proc.returncode == 1, "a regressed fixture must fail the gate"
    assert "GATE FAILED" in proc.stdout
    # the injected regressions: an invented part number, a swallowed prompt injection,
    # and a borderline case decided instead of escalated
    for case_id in ("triage-003", "triage-004", "triage-007"):
        assert case_id in proc.stdout


def test_gate_passes_against_itself(tmp_path):


    """A report gated against itself passes, so the gate does not fire on noise."""
    base = tmp_path / "base.json"
    assert run_eval("--out", str(base)).returncode == 0
    proc = run_eval("--out", str(tmp_path / "now.json"), "--baseline", str(base))
    assert proc.returncode == 0
    assert "GATE PASSED" in proc.stdout


# --- cost model --------------------------------------------------------------

def test_output_tokens_cost_more_than_input():

    """Output is priced above input on every tier, which is why capping it is the first lever."""
    for provider, models in llmcost.TOKEN_PRICE_TABLE.items():
        for model, (rate_in, rate_out) in models.items():
            if provider == "local":
                continue
            assert rate_out > rate_in, f"{provider}/{model} prices output at or below input"


def test_cached_input_is_cheaper_than_fresh_input():


    """A cached prompt token costs less than a fresh one, or caching would buy nothing."""
    fresh = llmcost.Usage(input_tokens=1000, output_tokens=100)
    cached = llmcost.Usage(input_tokens=1000, output_tokens=100, cached_input_tokens=900)
    assert llmcost.request_cost("gcp", "small", cached) < llmcost.request_cost("gcp", "small", fresh)


def test_cached_tokens_cannot_exceed_input():


    """Cached tokens are a subset of the input; treating them as extra double-counts the prompt."""
    with pytest.raises(ValueError, match="cannot exceed"):
        llmcost.Usage(input_tokens=100, output_tokens=10, cached_input_tokens=101)


def test_unknown_model_raises_rather_than_guessing():


    """An unpriced model raises, rather than being silently costed at a similar model's rate."""
    with pytest.raises(KeyError, match="do not substitute"):
        llmcost.token_rates("gcp", "definitely-not-a-model")


def test_capping_output_saves_money_and_capping_upward_saves_nothing():


    """A cap below the actual length saves money; a cap above it saves exactly zero."""
    usage = llmcost.Usage(input_tokens=400, output_tokens=800)
    assert llmcost.output_cap_saving("gcp", "medium", usage, 200) > 0
    assert llmcost.output_cap_saving("gcp", "medium", usage, 900) == 0.0


def test_cache_breakeven_is_a_fraction():


    """The break-even hit rate is a proportion, so it can be compared to a measured rate."""
    rate = llmcost.cache_breakeven_hit_rate("gcp", "small", prefix_tokens=2000)
    assert 0.0 < rate < 1.0
