# Lab 5 — Cost report

Provider `gcp` · instance `e2-standard-4` · 5.44 THB/hour

Every rate quoted here was pulled from the Cloud Billing Catalog API on 2026-09-07 for
asia-southeast1 and converted at 1 USD = 32.921586 THB. `python scripts/price_check.py`
re-pulls them and fails if any has drifted more than 1%.

## 1. Estimate, made before running
60.00 THB

This is the forecast for the whole system through the end of term: a weekly training job
on spot e2-standard-4, a staging endpoint torn down after each session, object storage for
the dataset and the DVC remote, and the container images.

**Stated plainly so the next section is readable:** the actual figure below covers only
what has been consumed so far. Labs 1 to 4 were executed with local compute against real
GCP storage and registry, and the managed-compute work is scheduled for the following
session. Comparing a full-term forecast against a partial-term actual would flatter the
forecast, so both scopes are named rather than reconciled silently.

## 2. Actual, from billing filtered by tag
0.93 THB

Measured from the resources themselves rather than from a billing export, because billing
export to BigQuery is not configured on this project and the Cloud Billing API does not
expose consumed cost without it. What exists:

| Resource | Measured | Rate | THB per month |
|---|---|---|---|
| GCS `gs://itcs355-6688010/itcs355` (DVC remote + probes) | 0.327 MB | 0.020 USD/GiB/mo | 0.0002 |
| Artifact Registry `asia-southeast1/itcs355` (training + serving images) | 302.1 MB | 0.100 USD/GiB/mo | 0.926 |
| Cloud Monitoring custom metrics | 7 series, 1 write each | within free allowance | 0.000 |
| Managed compute | none provisioned | — | 0.000 |

The registry is 99.9% of it, and 302 MB for two Python images is itself a finding: the
multi-stage build keeps the compiler toolchain out, and the two images share the pinned
base layer, so the second image costs almost nothing to store.

## 3. The gap
-59.07 THB (-98.5%)

The gap is the compute that has not run yet, and naming it is more useful than closing it:

* **Managed training** — the study ran locally. Twelve trials on Vertex at 7.076 THB/h
  would have been about 8 THB of compute plus per-job scheduling overhead, and Vertex bills
  a minimum job duration, so twelve short jobs cost more than one long one. That per-job
  floor is the single biggest thing a local rehearsal hides.
* **The endpoint** — nothing was provisioned, so nothing billed by the hour. A Vertex
  endpoint at 8.889 THB/h left running over a weekend is 640 THB, which is 80% of the whole
  term budget. This is why Lab 3's teardown warning is the loudest paragraph in the course.
* **Egress and operations** — absent from the estimate, as they always are. At this data
  volume they are rounding error; at production volume they are not.

The honest summary: the estimate is untested against a bill. It will be checked against
one, by tag, in the session that provisions managed compute.

## 4. Breakdown by component
| Component | THB to date | Notes |
|---|---|---|
| Training | 0.00 | local CPU; would be 7.076 THB/h on Vertex e2-standard-4 |
| Storage | 0.0002/mo | 0.327 MB in GCS, including the DVC remote |
| Serving | 0.00 | local uvicorn and containers; no managed endpoint provisioned |
| Pipeline | 0.00 | compiled to reports/pipeline-vertex.yaml, not submitted |
| Monitoring | 0.00 | 7 custom metric writes, within the free allowance |
| Registry | 0.93/mo | 302.1 MB of images |

## 5. Cost per 1,000 predictions
Measured throughput: 239.7 req/s (Lab 3, 4 vCPU with 8 workers, p95 93.9 ms)

| Utilisation | THB per 1,000 |
|---|---|
| 5% | 0.1262 |
| 25% | 0.0252 |
| 80% | 0.0079 |

**Which one to believe: 5%, and not even that.** 240 machines checked once a day is 240
predictions a day. The measured configuration sustains 239.7 per second. Real utilisation
is under 0.01%, so the 5% column is already two orders of magnitude optimistic, and a
per-prediction cost computed at 80% utilisation would be a number describing a system
nobody is running. The three columns are reported because the lab asks for three; the
useful conclusion is that per-prediction pricing is the wrong unit for this workload
entirely.

Below roughly **0.0015 req/s**, scheduled batch inference is cheaper than keeping
this endpoint warm. Against the real rate — 240 predictions a day is 0.0028 req/s — this
system sits within a factor of two of the break-even line, and on the batch side of it once
the endpoint is a managed one at 8.889 THB/h rather than the rate above. The endpoint earns
its keep as an interactive triage screen, not as a way to score 240 rows.

## 6. One optimisation applied
| | Before | After |
|---|---|---|
| Configuration | 4 vCPU, 4 uvicorn workers | 4 vCPU, 8 uvicorn workers |
| Throughput at concurrency 10 | 216.8 req/s | 239.7 req/s |
| p95 latency | 102.81 ms (misses the 100 ms target) | 93.93 ms (meets it) |
| THB per 1,000 at 5% utilisation | 0.1396 | 0.1262 |
| Hourly rate | 5.443 THB/h | 5.443 THB/h |

Doubling the worker count on the same instance met the latency target that the instance
had been missing, and cut cost per prediction by 10%, for no extra money at all. The
alternative — the next instance size up — would have cost 5.443 THB/h more for 1.45x the
throughput. Requests spend real time blocked on the GIL and on socket I/O, so more
processes than cores is the right call until the worker count stops helping.

**What it cost:** more workers means more copies of the model in memory, roughly 90 MB
each, and a longer cold start as each loads. On a scale-to-zero platform that trade would
be worse, because cold start is already 2.3 s.

A second optimisation, found while measuring rather than while optimising, is worth more
than either: the registered model carried `n_jobs=-1` into serving, where joblib's fan-out
cost 37 ms of the 45 ms budget for one row. Setting it to 1 at load took p50 from 39.4 ms
to 9.9 ms and quadrupled throughput on the same hardware. Nothing was bought.
