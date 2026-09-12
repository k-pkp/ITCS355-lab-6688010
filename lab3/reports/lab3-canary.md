# Lab 3 Task 4 — canary, detection, rollback

Incumbent (registry version 1) in slot **blue**, canary (registry version 3) in slot
**green**, split 90/10 by `loadtest/canary_proxy.py`. The detector is told the slot
names and nothing else.

## Timeline

```
  4750 requests ·   45.0s · QUALITY ALERT — windowed AUC blue 0.8580 vs green 0.8090; green is worse
       rollback · 2026-09-07T13:00:10 · weights {'blue': 90, 'green': 10} -> {'blue': 100, 'green': 0}
       post-rollback traffic · blue 1000 · green 0 (of 1000)
```

## Detection

- **Prediction distribution (no outcomes needed)**: never fired within the request budget.
- **Windowed ROC AUC (needs outcomes)**: fired after 4750 requests (45.0s of wall clock).

Traffic served during the canary: blue 4284, green 466.
Traffic served after rollback: blue 1000, green 0.

The decision log at `reports/canary-decisions.jsonl` carries a timestamped record of
every routed request and of the weight change itself.

## Caveat on the request counts

Traffic is replayed from held-out rows drawn with replacement. Repeating a row adds
no new information about the model, so the counts above are the optimistic end of
how quickly live traffic would settle the question.
