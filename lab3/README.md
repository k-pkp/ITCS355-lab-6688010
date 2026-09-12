# Lab 3 — Serving, Load Testing, and Rollback

> **Lab 3 of 5.** Self-contained: every command runs from inside this folder. Repository
> index: [`../README.md`](../README.md).

**Marks:** 8 · **CLO2, CLO3** · Passes when the endpoint meets *your own stated* p95 target
and the rollback evidence shows traffic genuinely moved.

---

## What this lab asked for, and where it is

| Task | Deliverable | Where |
|:--|:--|:--|
| 1. Inference service | 4 routes, Pydantic validation, structured logs, version in body | [`service/app.py`](service/app.py), [`service/schemas.py`](service/schemas.py) |
| 2. Deploy | `deploy()` / `invoke()` | [`cloudlayer/gcp.py`](cloudlayer/gcp.py) — implemented, **not yet run** |
| 3. Load test honestly | 3 concurrency levels, p50/p95/p99, breaking point, batch/payload/instance | [`reports/lab3-load.md`](reports/lab3-load.md) |
| 4. Canary and rollback | Detection from metrics, timestamped evidence | [`reports/lab3-canary.md`](reports/lab3-canary.md), [`reports/canary-decisions.jsonl`](reports/canary-decisions.jsonl) |
| 5. Cost per 1,000 | With method and utilisation assumption stated | [`reports/lab3-load.md`](reports/lab3-load.md) |

---

## The target, stated before measuring

**p95 < 100 ms for a single `/predict` at concurrency 10, error rate < 1%.**

It is written as a k6 threshold in [`loadtest/k6.js`](loadtest/k6.js), so it fails the load
test rather than living in a document, and the reasoning sits in the comment beside it: this
service backs a triage screen a technician refreshes while standing at a machine, and under
about 100 ms reads as instant to a person.

## Headline results

| workers | concurrency | rps | p50 | p95 | p99 | vs target |
|--:|--:|--:|--:|--:|--:|:--|
| 1 | 1 | 97.6 | 9.77 | 11.60 | 13.78 | PASS |
| 1 | 10 | 44.0 | 226.67 | 277.16 | 305.54 | MISS |
| 4 | 10 | 230.7 | 34.20 | 93.57 | 115.98 | PASS |
| 8 | 10 | 365.2 | 20.53 | 58.33 | 77.52 | PASS |
| 8 | 50 | 260.9 | 183.86 | 354.35 | 476.80 | MISS |

**Breaking concurrency:** 4 for one worker, 15 for eight. Error rate stayed at 0% at every
level measured — this service degrades by queueing, not by failing, which is the more
dangerous shape because nothing alerts.

**Two bugs found by measuring, not by reading:**

1. Throughput was pinned at 27 rps at concurrency 1, 10 and 50 alike. Flat throughput under
   rising load means the bottleneck is inside one request. The registered model carried
   `n_jobs=-1` from training, so scoring a single row made joblib fan 200 trees across 20
   workers: 37.4 ms, of which 8.0 ms was the forest. Fixed at the load point in
   `service.app.tune_for_single_row_serving`. p50 fell 39.4 → 9.9 ms.
2. k6 reported p50 51.9 ms while the service's own logs said 9.8 ms for the same requests.
   Keep-alive against a multi-worker uvicorn adds a flat ~40 ms delayed-ACK stall. Every
   number above is measured with `K6_NO_CONNECTION_REUSE=true`; a real client that pools
   connections needs `TCP_NODELAY`, which is worth knowing before blaming the model.

**Batching:** one `/predict/batch` call with 100 rows takes 11.98 ms against 1084.93 ms for
100 separate calls — **90.5x cheaper per row**.

**Cost per 1,000 predictions**, at the verified rates, three utilisations:

| configuration | THB/h | rps | 5% | 25% | 80% |
|:--|--:|--:|--:|--:|--:|
| Cloud Run, 1 vCPU / 2 GiB | 3.129 | 40.4 | 0.4303 | 0.0861 | 0.0269 |
| e2-standard-4, 4 vCPU / 8 workers | 5.443 | 239.7 | 0.1262 | 0.0252 | 0.0079 |
| Vertex endpoint, n1-standard-4 | 8.889 | 239.7 | 0.2060 | 0.0412 | 0.0129 |

The utilisation assumption is where the number is most fragile, and the honest reading is
that none of the three columns applies: 240 machines checked daily is 0.0028 req/s against a
configuration that sustains 239.7. Below **0.0025 req/s — about 213 predictions a day** —
scheduled batch is cheaper than a warm endpoint, so this workload sits on the batch side of
the line and the endpoint exists for the interactive screen, not for the volume.

## Canary and rollback

Version 1 (test ROC AUC 0.8499) in slot **blue**, version 3 (0.8341) in slot **green**, split
90/10 by [`loadtest/canary_proxy.py`](loadtest/canary_proxy.py). The detector is given slot
names and nothing else — never which slot holds the new model.

- Detected at **4,750 requests, 45 s**, by windowed ROC AUC per slot: blue 0.8580 against
  green 0.8090, held for three consecutive checks.
- Rolled back at `2026-09-07T13:00:10`, `{"blue": 90, "green": 10} -> {"blue": 100, "green": 0}`.
- The 1,000 requests after the change went **blue 1,000 / green 0**.

The label-free signal — mean predicted probability per slot — separated the two by 0.008
(blue 0.1140, green 0.1060) and never fired at the 0.05 threshold set in advance. That is the
uncomfortable part: the degradation was invisible in what the model emitted, and only
outcomes revealed it. In the real system those outcomes take seven days.

---

## Reproduce it

```bash
make data
python scripts/export_model.py --out reports/model.joblib
MODEL_PATH=reports/model.joblib MODEL_VERSION=local \
  uvicorn service.app:app --port 8080 --workers 8      # in one terminal

K6_NO_CONNECTION_REUSE=true k6 run \
  --summary-trend-stats="avg,min,med,max,p(50),p(95),p(99)" \
  -e TARGET=http://127.0.0.1:8080/predict -e VUS=10 -e DURATION=15s loadtest/k6.js

python loadtest/batch_vs_single.py --target http://127.0.0.1:8080
pytest -q tests/test_service.py tests/test_integration_container.py
```

The canary needs three processes — two model versions and the proxy — and the exact
sequence is in [`reports/lab3-canary.md`](reports/lab3-canary.md).

## What is implemented but not yet run

`deploy()` and `invoke()` target Vertex AI endpoints, including the route configuration in
[`cloudlayer/routes.py`](cloudlayer/routes.py) that absorbs each provider's expected paths
without renaming the service's own routes. No managed endpoint was provisioned, so nothing
billed by the hour. A Vertex endpoint at 8.889 THB/h left over a weekend is 640 THB, which is
80% of the term budget — the reason the teardown warning is the loudest paragraph in the lab.
