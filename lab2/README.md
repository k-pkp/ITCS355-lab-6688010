# Lab 2 — Experiment Tracking and Model Registry

> **Lab 2 of 5.** Self-contained: every command runs from inside this folder. Repository
> index: [`../README.md`](../README.md).

**Marks:** 8 · **CLO1, CLO3** · The lab grades the *justification*, not the metric.

---

## What this lab asked for, and where it is

| Task | Deliverable | Where |
|:--|:--|:--|
| 1. Managed training | `submit_training()` / `wait_training()` | [`cloudlayer/gcp.py`](cloudlayer/gcp.py) — implemented, **not yet run** (see below) |
| 2. Budgeted study | 12+ trials, 3+ hyperparameters, cost per trial | [`src/tune.py`](src/tune.py) · `make tune` |
| 3. Compare and justify | Comparison artifact + 200-word justification | [`reports/lab2-comparison.md`](reports/lab2-comparison.md) |
| 4. Register with lineage | 8 lineage fields on the *version* | [`scripts/register_model.py`](scripts/register_model.py) · `make register` |
| 5. Prove reload | Load by version, from the registry | [`scripts/reload_check.py`](scripts/reload_check.py) · `make reload-check` |

---

## What was done

**The study: 12 trials over five hyperparameters.** `n_estimators`, `max_depth`,
`min_samples_leaf`, `max_features` and `class_weight` — chosen because each changes a
different property of the model rather than being another way to say "bigger forest".

One thing worth pointing at in `src/tune.py`: the scaffold sliced the first N entries off
`itertools.product`, which holds the leading hyperparameters at their first value and varies
only the trailing ones — a study that looks like five dimensions and is really two.
`choose_trials()` shuffles the grid with the seed first, so 12 trials actually spread across
all five, and still reproduce exactly on a rerun.

**Prices were verified, not assumed.** [`scripts/price_check.py`](scripts/price_check.py)
re-pulls every GCP rate in [`src/costs.py`](src/costs.py) from the Cloud Billing Catalog API
— the same SKUs the console pricing page renders — for `asia-southeast1`, and fails if any
has drifted more than 1%.

```
$ make price-check
  instance                          live THB/h  in costs.py  status
  e2-standard-4                          5.443        5.443  ok
  n1-standard-4                          7.716        7.716  ok
  vertex-training-e2-standard-4          7.076        7.076  ok
  vertex-endpoint-n1-standard-4          8.889        8.889  ok
  cloud-run-1vcpu-2gib                   3.129        3.129  ok

  spot discount factors (measured, not assumed)
  e2-standard-4                         0.5455       0.5455
  n1-standard-4                         0.2678       0.2678
```

That last block is a finding. The course scaffold said discounted compute is "roughly 30% of
on-demand across all three providers". On GCP in Singapore that is true of N1 (0.268) and
wrong by a factor of two for E2 (0.546). Spot discounts are set per machine family and per
region, so one global constant quietly halves or doubles an estimate.

**The model that was registered is not the highest-scoring one.** Full argument in the
report; the short version is that the 0.0047 gap between them is smaller than the 0.0106
seed noise, while the price difference — 2.5x — is real and repeats on every retrain.

**Lineage, all eight fields, set on the version rather than the run:**

```json
{
  "git_commit":      "8512d6e21c83e2fa071c30d894cf4edaab99cc3c",
  "data_version":    "1c886b512c8a5c9bf723da1cd119fc80.dir",
  "mlflow_run_id":   "61b401bc27b0454b9baa2711ff583c55",
  "training_job_id": "local://keng/20260907T123535",
  "image_digest":    "…/itcs355-lab1@sha256:f43b61cbecc7acfb04cd18125e72ebaa79d4571f7bf6bf58ff32a1d0da81d838",
  "seed":            "20260101",
  "metric_val":      "0.841635",
  "metric_test":     "0.849892"
}
```

Tags on the run would have been the near-miss: the run and the registered version are
different objects, and it is the version a future engineer starts from.

**Reload from the registry works** — `scripts/reload_check.py` pulls `models:/…/1` by
version and scores five held-out rows. This is the lab's quiet test, and the one that
catches serialization assumptions six months later.

---

## Reproduce it

```bash
make data                                   # generate the dataset
make tune                                   # 12 trials, budget enforced
make compare                                # writes reports/lab2-comparison.md
make register                               # registers with all eight lineage fields
make reload-check MODEL_REGISTRY_NAME=itcs355-6688010 VERSION=1
make price-check                            # re-verify every rate against the live catalog
```

## What is implemented but not yet run

`submit_training()` and `wait_training()` target Vertex AI custom training jobs. The Vertex
API is not enabled on this project yet, so the study ran on local CPU priced at the verified
`e2-standard-4` rate, and every cost figure in the report is labelled as modelled rather
than billed. The commands to run it for real are in the repository index.
