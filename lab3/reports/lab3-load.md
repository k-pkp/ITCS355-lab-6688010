# Lab 3 — serving, load, and rollback

Everything below was measured on 2026-09-07 against the registered model (MLflow registry
`itcs355-6688010`, version 1) served by `service/app.py`. Load generator: k6 v2.2.0,
`loadtest/k6.js`, committed. Host: WSL2, 20 logical cores; container runs are capped with
`docker run --cpus` so the shapes below correspond to instances that can actually be bought.

## The target, stated before measuring

**p95 < 100 ms for a single `/predict` at concurrency 10, error rate < 1%.**

Written into `loadtest/k6.js` as a k6 threshold before the first run, so it fails the load
test rather than living in a document. The reasoning is in the comment there: this service
backs a maintenance triage screen a technician refreshes while standing at a machine.
Under about 100 ms reads as instant to a person, so latency below that buys nothing anyone
notices and the budget is better spent on a smaller instance.

## Two measurement artefacts found before any conclusions were drawn

**1. The model was 4x slower than its own arithmetic.** The first run showed throughput
pinned at 27 requests per second at concurrency 1, 10 and 50 alike — flat throughput under
rising load means the bottleneck is inside one request, not in the load. The registered
estimator carries `n_jobs=-1` from training, so scoring a single row made joblib fan 200
trees across 20 workers: 37.4 ms, of which 8.0 ms was the forest. Fixed at the load point
in `service.app.tune_for_single_row_serving`, which sets `n_jobs=1` when the model is
loaded for serving. p50 at concurrency 1 fell from 39.4 ms to 9.9 ms.

**2. Keep-alive against a multi-worker server added a flat ~40 ms.** With four uvicorn
workers, k6 reported p50 51.9 ms at concurrency 1 while the service's own logs said 9.8 ms
and `curl` said 10.6 ms for the same requests. Re-running with
`K6_NO_CONNECTION_REUSE=true`: 10.4 ms. It is a Nagle / delayed-ACK interaction between the
load generator's pooled connections and the shared listening socket, and it is a property
of the client, not the service. Every number below is measured with connection reuse
disabled; a real client that reuses connections would need `TCP_NODELAY` set, which is
worth knowing before blaming the model.

## Concurrency levels

Host process, no CPU cap, model loaded from the registry.

| workers | concurrency | throughput (rps) | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | errors | vs target |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 | 1  | 97.6  | 9.77 | 11.60 | 13.78 | 18.56 | 0% | PASS |
| 1 | 10 | 44.0  | 226.67 | 277.16 | 305.54 | 340.61 | 0% | MISS |
| 1 | 50 | 44.4  | 1159.56 | 1331.70 | 1404.89 | 1478.15 | 0% | MISS |
| 4 | 1  | 96.2  | 9.84 | 11.92 | 14.84 | 24.30 | 0% | PASS |
| 4 | 10 | 230.7 | 34.20 | 93.57 | 115.98 | 176.48 | 0% | PASS |
| 4 | 50 | 183.4 | 263.59 | 451.78 | 500.00 | 581.97 | 0% | MISS |
| 8 | 1  | 93.6  | 10.04 | 12.57 | 14.77 | 26.39 | 0% | PASS |
| 8 | 10 | 365.2 | 20.53 | 58.33 | 77.52 | 112.97 | 0% | PASS |
| 8 | 50 | 260.9 | 183.86 | 354.35 | 476.80 | 737.47 | 0% | MISS |

k6 does not export p99 by default. It is measured here by asking for it explicitly, which
is the whole command:

```bash
K6_NO_CONNECTION_REUSE=true k6 run --summary-trend-stats="avg,min,med,max,p(50),p(95),p(99)" \
  -e TARGET=http://127.0.0.1:8080/predict -e VUS=10 -e DURATION=15s loadtest/k6.js
```

**What the p99 column says that p95 does not.** At concurrency 1 the gap between p95 and
p99 is 2 ms — the distribution has no tail, which is the signature of a service doing the
same fixed amount of work every time. The gap widens with load (58 → 78 ms at 8 workers,
94 → 116 ms at 4) but stays proportional; it never blows out. That matters because the
classic p99 pathology this course warns about — a p99 far above p95 in a repeating pattern —
means a cold start or a model reload per request, and this shape rules both out. The model
is loaded once at startup, and these numbers are the evidence rather than the claim.

**Breaking concurrency.** One worker crosses the 100 ms target between concurrency 3
(p95 80.3 ms) and 4 (p95 108.1 ms). Eight workers cross between 10 (58.2 ms) and 15
(101.6 ms): 20 → 153.3 ms, 25 → 182.5 ms, 30 → 219.5 ms. Error rate stayed at 0% at every
level measured — this service degrades by queueing, not by failing, which is the more
dangerous failure mode because nothing alerts.

## Instance size

Containerised, `--cpus` enforced, so these are the shapes on the price list.

| vCPU | workers | concurrency 10 rps | p95 (ms) | vs target |
|--:|--:|--:|--:|:--|
| 1 | 1 | 40.4  | 321.58 | MISS |
| 2 | 2 | 88.8  | 194.81 | MISS |
| 4 | 4 | 216.8 | 102.81 | MISS (by 3%) |
| 4 | 8 | 239.7 | 93.93  | PASS |
| 8 | 8 | 348.7 | 57.42  | PASS |

The cheapest configuration that meets the stated target is **4 vCPU with 8 workers**, and
the second row of that pair is the finding: at a fixed 4 vCPU, doubling workers from 4 to 8
moved p95 from 102.8 ms to 93.9 ms and throughput from 216.8 to 239.7 rps, at no extra
cost at all. Requests spend real time blocked on the GIL and on socket I/O, so more
processes than cores is the right call here. Buying the next instance size up would have
cost 5.443 THB/h more and delivered the same pass.

**Cost change for one step up**, at the verified asia-southeast1 rates in `src/costs.py`:
e2-standard-4 is 5.443 THB/h; the next step up doubles that to about 10.9 THB/h for
roughly 1.45x the throughput (239.7 → 348.7 rps). Scaling out is worse value than scaling
the worker count, until the worker count stops helping.

**Cold start**, measured three times on a 1-vCPU container: 2.44 s, 2.25 s, 2.33 s from
`docker run` to `/ready` returning 200, with the first prediction served 20 ms later. On a
scale-to-zero platform that is the p99 for the first request after an idle period, and it
is 24x the entire latency budget. It is the reason the scale-to-zero option below is not
automatically the right answer.

## Batching

`loadtest/batch_vs_single.py`, 100 rows, median of 5 repeats:

| approach | total | per row |
|:--|--:|--:|
| 100 separate `/predict` calls | 1084.93 ms | 10.849 ms |
| one `/predict/batch` call | 11.98 ms | 0.120 ms |

**Batching is 90.5x faster per row.** Almost all of the per-request cost is fixed —
connection, parse, validate, build a DataFrame, dispatch — and the forest itself is
cheap once it is scoring more than one row.

## Payload size

| rows | body bytes | total ms | ms per row |
|--:|--:|--:|--:|
| 1   | 160    | 10.77 | 10.768 |
| 10  | 1,517  | 10.98 | 1.098 |
| 25  | 3,768  | 11.89 | 0.476 |
| 50  | 7,521  | 11.75 | 0.235 |
| 100 | 15,035 | 11.53 | 0.115 |

Inside the schema's 100-row cap, serialization never dominates: the total barely moves
from 1 row to 100. Measured off the wire to find where it eventually would — JSON parse
versus DataFrame build versus scoring:

| rows | json parse | frame build | scoring | serialization share |
|--:|--:|--:|--:|--:|
| 100     | 0.06 ms | 3.82 ms  | 8.58 ms   | 0.5% |
| 1,000   | 0.72 ms | 1.22 ms  | 10.06 ms  | 6.0% |
| 10,000  | 6.03 ms | 4.54 ms  | 23.54 ms  | 17.7% |
| 100,000 | 75.04 ms| 32.63 ms | 171.63 ms | 26.9% |

Serialization becomes material around 10,000 rows and is still not the largest term at
100,000. The binding constraint on this API is the 100-row cap in `service/schemas.py`,
not the wire format.

## Cost per 1,000 predictions

Method: `src.costs.cost_per_1k_predictions` — hourly rate ÷ (throughput × utilisation) ×
1000 seconds-equivalent. Rates verified against the Cloud Billing Catalog API for
asia-southeast1 (`scripts/price_check.py`). Throughput is the measured sustained figure
for the configuration, not a peak.

| configuration | THB/h | rps | 5% util | 25% util | 80% util |
|:--|--:|--:|--:|--:|--:|
| Cloud Run, 1 vCPU / 2 GiB, 1 worker | 3.129 | 40.4 | 0.4303 | 0.0861 | 0.0269 |
| e2-standard-4 rate, 4 vCPU / 8 workers | 5.443 | 239.7 | 0.1262 | 0.0252 | 0.0079 |
| Vertex AI endpoint, n1-standard-4 | 8.889 | 239.7 | 0.2060 | 0.0412 | 0.0129 |

**The utilisation assumption is where this number is most fragile.** 240 requests per
second is 20.7 million predictions a day; a maintenance team of a dozen technicians will
not generate 1% of that. At the honest utilisation for this system the per-1,000 figure is
meaningless, because the endpoint is being paid for by the hour to sit idle. Which leads
directly to:

**When is batch cheaper than a warm endpoint?** Assuming one batch run a day on spot
e2-standard-4 for 10 minutes (0.4949 THB) against a Vertex endpoint at 8.889 THB/h
(213.3 THB/day): batch wins below **0.0025 requests per second, about 213 predictions a
day**. Above that the warm endpoint is cheaper per prediction. For 240 machines checked
once a day this system is on the batch side of that line, and the endpoint exists for the
interactive triage screen, not for the volume.

## Canary and rollback

Full timeline and evidence in `reports/lab3-canary.md`; decision log with per-request
timestamps in `reports/canary-decisions.jsonl`.

Setup: registry version 1 (test ROC AUC 0.8499) in slot **blue**, registry version 3 in
slot **green** (test ROC AUC 0.8341 — the shape of a real regression: depth limit removed
and `class_weight="balanced"` switched on "to help with the imbalance"). Split 90/10 by
`loadtest/canary_proxy.py`. The detector is given slot names only, never versions.

**Five lines.**

1. *What metric revealed it:* windowed ROC AUC computed per slot over a 1,500-request
   window — blue 0.8580 against green 0.8090, held for three consecutive checks.
2. *How long detection took:* 4,750 requests, 45 seconds of wall clock. Green served 466
   of those; the wait was for the canary's own 10% to accumulate enough outcomes.
3. *What would have made it faster:* nothing available at prediction time would have.
   The label-free signal — mean predicted probability per slot — separated the two by
   0.008 (blue 0.1140, green 0.1060), far inside noise and nowhere near the 0.05 threshold
   set in advance; it never fired. What would genuinely help is a larger canary share, and
   a metric that needs no outcomes but is sharper than the mean, such as the KS distance
   between each slot's probability distribution and a stored reference.
4. *What would have happened at 50/50:* green would have accumulated outcomes five times
   faster, so the same alert would have fired at roughly 950 requests instead of 4,750 —
   at the price of serving the worse model to half of all traffic while waiting.
   90/10 buys blast radius with detection time, and this is what that trade costs.
5. *Evidence traffic moved:* the weight change is logged at `2026-09-07T13:00:10` with
   `{"blue": 90, "green": 10} -> {"blue": 100, "green": 0}`, and the 1,000 requests sent
   after it went blue 1,000 / green 0.

**The uncomfortable part.** Outcomes here are simulated as arriving instantly. In the real
system `failed_within_7d` takes seven days, so the quality signal that caught this would
have been a week late, and the label-free signal that was available immediately could not
see the problem at all. A canary on a model whose labels are that delayed is a much weaker
control than it looks.
