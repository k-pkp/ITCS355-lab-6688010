# Lab 2 — Run comparison

Experiment `itcs355-lab2` · 12 trials · total spend 0.0172 THB

`thb_per_point` is cost per percentage point of val_roc_auc above the worst trial. Cheap improvements rank low; expensive improvements rank high, however good the headline number is.

| run_id   |   val_roc_auc |   cost_thb |   n_estimators | max_depth   |   min_samples_leaf | max_features   | class_weight   |   thb_per_point |
|:---------|--------------:|-----------:|---------------:|:------------|-------------------:|:---------------|:---------------|----------------:|
| 3843d0db |        0.8463 |     0.0015 |            500 | 10          |                 20 | sqrt           | None           |          0.0005 |
| a428b2e6 |        0.8446 |     0.0018 |            500 | 6           |                  1 | sqrt           | balanced       |          0.0007 |
| b6dc2ba6 |        0.8416 |     0.0006 |            200 | 6           |                  5 | sqrt           | None           |          0.0003 |
| 8fd55210 |        0.8409 |     0.0016 |            500 | 6           |                 20 | 0.5            | balanced       |          0.0007 |
| 9438d12e |        0.8379 |     0.0016 |            500 | 10          |                 20 | 0.5            | balanced       |          0.0008 |
| 446d8d41 |        0.8378 |     0.0015 |            500 | 6           |                  5 | 0.5            | None           |          0.0008 |
| ad443c5f |        0.8271 |     0.0017 |            500 | 10          |                  5 | 0.5            | balanced       |          0.0018 |
| e0f7d6f2 |        0.8264 |     0.0019 |            500 | None        |                  5 | 0.5            | None           |          0.0022 |
| c956b8b8 |        0.8242 |     0.002  |            500 | None        |                  1 | sqrt           | None           |          0.0032 |
| c61838d3 |        0.8223 |     0.0007 |            200 | None        |                  5 | 0.5            | balanced       |          0.0016 |
| 03ec3e21 |        0.8202 |     0.0016 |            500 | 10          |                  1 | 0.5            | None           |          0.0069 |
| 1af309e9 |        0.8179 |     0.0007 |            200 | None        |                  1 | sqrt           | balanced       |          0.5357 |

## Which model did you register, and why?

Registered: 200 trees, max_depth 6, min_samples_leaf 5, max_features sqrt, no class
weighting — trial `b6dc2ba6`, val_roc_auc 0.8416. The best trial scored 0.8463.

**Why not the highest scorer.** The 0.0047 gap is smaller than the noise. Across five
seeds the top configuration averages val 0.8577 (sd 0.0127) and the registered one 0.8556
(sd 0.0138); the intervals overlap almost completely, so the ranking between them is a
coin toss re-flipped by the split. What is not noise is the price: 500 trees cost 0.0015
THB per fit against 0.0006, 2.5x for a difference we cannot measure, and that multiple
follows the model into every retrain and every prediction.

**Seed variance.** val 0.8556 ± 0.0138, test 0.8540 ± 0.0087 over seeds
20260101–20260105. Any claim finer than the second decimal place is unsupported.

**Cost.** 0.0006 THB per fit at the verified e2-standard-4 rate (5.443 THB/h,
asia-southeast1). Weekly retraining is about 0.03 THB a month of compute; the pipeline
around it, not the fit, is what costs money.

**How this could be wrong.** Sixty machines per split is thin. If the real cohort is
larger and more varied, depth 6 will underfit where depth 10 would not, and the seed
noise that hides the difference here would shrink until it is real.