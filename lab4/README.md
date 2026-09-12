# Lab 4 — CI/CD, Observability, and Drift

> **Lab 4 of 5.** Self-contained: every command runs from inside this folder. Repository
> index: [`../README.md`](../README.md).

**Marks:** 8 · **CLO2, CLO3** · Passes when the deliberately bad commit is blocked by a
*named* test and the injected drift fires a real alert with reasoning that holds up.

---

## What this lab asked for, and where it is

| Task | Deliverable | Where |
|:--|:--|:--|
| 1. Four kinds of test | unit · data contract · model behaviour · integration | [`tests/`](tests) |
| 2. CI pipeline | lint → unit → contract → behaviour → build → integration | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| 3. Bad commit blocked | The failing run and the test that caught it | [`reports/lab4-blocked-commit.md`](reports/lab4-blocked-commit.md) |
| 4. Dashboard + SLO | 5 signals, and an error-budget response | [`monitoring/dashboard.json`](monitoring/dashboard.json), [`monitoring/slo.yaml`](monitoring/slo.yaml) |
| 5. Drift on a schedule | Detector, justified threshold, metric emission | [`monitoring/drift.py`](monitoring/drift.py) |
| 6. Injected drift | Alert, timings, five-line post-mortem | [`reports/lab4-drift.md`](reports/lab4-drift.md), [`reports/lab4-postmortem.md`](reports/lab4-postmortem.md) |

---

## The four test categories, and what each protects against

| Category | File | What failing it means |
|:--|:--|:--|
| **Unit** | [`tests/test_drift.py`](tests/test_drift.py) | The PSI/KS maths changed, so the alerting threshold no longer means what it was measured to mean. No network, no model, no data file. |
| **Data contract** | [`tests/test_data.py`](tests/test_data.py) | An upstream producer changed something, even though our code is untouched. |
| **Model behaviour** | [`tests/test_model_behaviour.py`](tests/test_model_behaviour.py) | The model does something the domain forbids — a healthy machine scoring high, risk falling as wear rises, a constant predictor, a row taking longer than its share of the latency budget. |
| **Integration** | [`tests/test_integration_container.py`](tests/test_integration_container.py) | The *image* is broken even though the app is fine: a dependency left in the build stage, a non-root user that cannot read what it needs, a port that does not serve. |

The incident each data contract test would have caught is named in the repository index —
that mapping is a graded deliverable and it is written out there in full.

## The drift threshold, and why it is 0.20

Not a library default. Two measurements bracket it:

- **Noise floor.** PSI between the reference and a 600-row window of *unchanged* data peaked
  at **0.043** over 200 draws across all six features. Below that is the window size talking.
- **Harm floor.** Shifting the most important feature by one standard deviation reaches PSI
  1.03 and costs 0.0007 ROC AUC — nothing. It takes a two-sigma shift, **PSI 3.54**, before
  the metric moves by 0.011.

0.20 is 4.6x the noise ceiling and an order of magnitude below the harm floor, deliberately:
the label here arrives seven days late, so the input-side statistic is the only same-day
signal, and its job is to open an investigation rather than trigger a retrain.

## The injected drift exercise

| mode | what changed | top PSI | alert | ROC AUC change |
|:--|:--|--:|:--|--:|
| shift | `temp_c` mean +6 °C | 0.3833 | **fired** | −0.0100 |
| scale | `vibration_mm_s` spread ×1.6, mean unchanged | 0.2627 | **fired** | −0.0167 |
| mix | 20% of machines over-represented 5:1 | 0.0087 | silent | +0.0019 |

Three findings, none of which is "the detector works":

1. **PSI ranked the two alerts in the wrong order.** The shift scored higher and cost less.
   PSI measures how far the input moved, not how much the model minds.
2. **The scale case moved the mean not at all** — 4.5869 before and after — and did the most
   damage. A monitor watching feature averages would have drawn a flat line through the worst
   of the three.
3. **The realistic drift was invisible.** A fleet composition change is what actually happens
   when a customer adds machines, and per-feature PSI could not see it at 0.0087. Catching it
   needs a multivariate statistic or a per-segment metric, and this detector has neither.

Detection latency was 0.72 s unscheduled. On the weekly schedule the honest number is **up to
7 days plus 0.72 s**, mean 3.5 days — if drift detection is meant to be a control rather than
a report, the schedule is the thing to change, not the code.

## The blocked bad commit

One line added to `scripts/make_dataset.py` on branch `demo/break-data-contract`:

```python
df = df.drop(columns=["ambient_humidity"])
```

It carries 4.9% of the model's importance and reads noisy, so dropping it from the export
looks like tidying. CI stops at the data contract step:

```
E       AssertionError: missing columns: ['ambient_humidity']
tests/test_data.py:35: AssertionError
3 failed, 7 passed in 0.59s
```

Three tests failed and the order matters: the schema test *names* the problem, the other two
fail with a bare `KeyError` because they look up a column that is not there. A suite where
only the `KeyError`s fired would tell you something broke without telling you what. Nothing
downstream runs — `build` needs `test`, so no image is built and CD never triggers.

---

## Reproduce it

```bash
make data
pytest -q tests/                                    # all four categories
make inject-drift && make drift                     # shift mode; exits 2 on alert
python scripts/inject_drift.py --mode scale --feature vibration_mm_s --magnitude 1.6 \
  --out data/current-scale.csv && python -m monitoring.drift --current data/current-scale.csv
python -m monitoring.drift --current data/current.csv --emit   # writes to Cloud Monitoring
```

`emit_metric` was verified against real Cloud Monitoring on 2026-09-07; the series lands
under `custom.googleapis.com/itcs355/drift.psi.<feature>`, which is exactly what the
dashboard panel queries.

## Note on the workflows

`.github/` is duplicated here so the files can be read alongside the rest of Lab 4. GitHub
only executes workflows found at the **repository root**, so the copy that actually runs is
`../.github/`. They are identical.
