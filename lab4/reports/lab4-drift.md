# Lab 4 — drift detection, threshold, and the injected-drift exercise

Detector: `monitoring/drift.py`, PSI and two-sample Kolmogorov–Smirnov implemented
directly. Reference window: `data/raw/sensors.csv`, 6,000 readings from 240 machines.
Threshold: **PSI ≥ 0.20**, defined and defended at `PSI_ALERT` in that file.

## How the threshold was chosen

Not from the library default. Two measurements bracket the choice.

**Noise floor — what "nothing happened" looks like.** PSI between the reference and a
600-row window drawn from the same unchanged data, 200 draws per feature:

| feature | max PSI | p95 PSI |
|:--|--:|--:|
| temp_c | 0.0432 | 0.0292 |
| vibration_mm_s | 0.0400 | 0.0258 |
| pressure_kpa | 0.0302 | 0.0220 |
| hours_since_service | 0.0355 | 0.0272 |
| load_pct | 0.0326 | 0.0256 |
| ambient_humidity | 0.0311 | 0.0251 |

Nothing unchanged exceeded 0.044. A threshold below about 0.05 would alert on the window
size.

**Harm floor — what the model actually notices.** Shifting `vibration_mm_s`, which carries
41% of the model's importance, and measuring test ROC AUC:

| shift | PSI | ROC AUC | drop |
|:--|--:|--:|--:|
| 0.10 sd | 0.0185 | 0.8501 | −0.0002 |
| 0.25 sd | 0.0749 | 0.8524 | −0.0026 |
| 0.50 sd | 0.2498 | 0.8511 | −0.0012 |
| 1.00 sd | 1.0294 | 0.8506 | −0.0007 |
| 2.00 sd | 3.5425 | 0.8393 | +0.0106 |

Ranking is remarkably robust to a uniform shift — it takes PSI above 3 before ROC AUC
moves by a hundredth. So there is a wide band between "not noise" (0.05) and "actually
harmful" (about 2.5), and the threshold is a choice about which end to sit at.

**0.20 is 4.6x the noise ceiling and an order of magnitude below the harm floor, on
purpose.** The label here — `failed_within_7d` — arrives a week late, so the input-side
statistic is the only signal available on the day something changes. Its job is to open an
investigation, not to trigger a retrain.

## The injected-drift exercise

`make inject-drift` and two variants, each scored immediately afterwards.

| mode | what changed | top PSI | KS | alert | ROC AUC change |
|:--|:--|--:|--:|:--|--:|
| shift | `temp_c` mean +6 °C | 0.3833 (temp_c) | 0.2457 | **fired** | −0.0100 |
| scale | `vibration_mm_s` spread x1.6, mean unchanged | 0.2627 (vibration_mm_s) | 0.1310 | **fired** | −0.0167 |
| mix | 20% of machines over-represented 5:1 | 0.0087 (temp_c) | 0.0342 | silent | +0.0019 |

Three findings, and none of them is "the detector works".

1. **PSI ranked the two alerts in the wrong order.** The shift scored higher (0.383 against
   0.263) but cost less accuracy (−0.0100 against −0.0167). PSI measures how far the input
   moved, not how much the model minds, and those are different questions.
2. **The scale case moved the mean not at all** — 4.5869 before, 4.5869 after — and was
   still the most damaging of the three. A monitor watching feature averages would have
   seen a flat line through the worst of the three incidents. This is the case for PSI over
   a mean, and it is why the drift panel plots PSI.
3. **The realistic drift was invisible.** The `mix` mode is a fleet composition change — the
   thing that actually happens when a customer adds machines or a site goes offline — and
   per-feature PSI could not see it at 0.0087, below the noise floor. It also did no harm
   here, which is luck rather than reassurance: a composition change that favoured machines
   the model handles badly would be equally invisible and would hurt. Catching that needs a
   multivariate statistic or a per-segment metric, and this detector has neither.

**Detection latency.** Injection to alert was 0.72 s, 0.68 s and 0.72 s unscheduled — the
detector itself is not the bottleneck. On the weekly schedule in `pipeline/pipeline.yaml`
the honest number is up to 7 days plus 0.72 s, and the mean is 3.5 days. If drift detection
is meant to be a control rather than a report, that schedule is the thing to change, not
the code.

**Metric emission.** `python -m monitoring.drift --emit` writes each feature's PSI to Cloud
Monitoring as `custom.googleapis.com/itcs355/drift.psi.<feature>` through
`GcpAdapter.emit_metric`. Verified by a real write on 2026-09-07; the dashboard panel in
`monitoring/dashboard.json` queries that prefix and draws the 0.20 threshold as a red line
so the panel and the alert cannot disagree.

---

## Post-mortem

The five-line post-mortem for this incident is a separate document:
**[`reports/lab4-postmortem.md`](lab4-postmortem.md)**.
