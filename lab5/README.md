# Lab 5 — Operating LLM Systems and Defending the Bill

> **Lab 5 of 5.** Self-contained: every command runs from inside this folder. Repository
> index: [`../README.md`](../README.md).

**Marks:** 8 · **CLO2, CLO3** · Two halves: Part A defends the bill (Tasks 1–6), Part B
operates an LLM step (Tasks 7–9).

---

## What this lab asked for, and where it is

| Task | Deliverable | Where |
|:--|:--|:--|
| 1. Managed pipeline with a real gate | DAG + compiler + a reachable "do not register" branch | [`pipeline/pipeline.yaml`](pipeline/pipeline.yaml), [`cloudlayer/pipelines.py`](cloudlayer/pipelines.py), [`reports/pipeline-vertex.yaml`](reports/pipeline-vertex.yaml) |
| 2. Least privilege | Three identities, and evidence of what broke | [`reports/lab5-portability.md`](reports/lab5-portability.md) |
| 3. Scheduled retraining | Three-strategy comparison for *this* system | repository index |
| 4. Portability test | Audit, swap, and an honest verdict | [`reports/lab5-portability.md`](reports/lab5-portability.md) |
| 5. Cost accounting | Six sections, three utilisations, one applied optimisation | [`reports/lab5-cost.md`](reports/lab5-cost.md) |
| 6. Teardown, verified | Tagged deletion and a re-check | [`scripts/teardown_verify.py`](scripts/teardown_verify.py) |
| 7. Eval gate that can fail | 10+ golden cases, a baseline, a failing run | [`evals/golden/maintenance-triage.jsonl`](evals/golden/maintenance-triage.jsonl), [`reports/lab5-llm.md`](reports/lab5-llm.md) |
| 8. Guardrail demonstrated | The failure, the control, the evidence | [`reports/lab5-llm.md`](reports/lab5-llm.md) |
| 9. Token bill | Three figures with working, plus the budget defence | [`reports/lab5-llm.md`](reports/lab5-llm.md) |

---

## The pipeline gate is a real branch

`make pipeline` compiles [`pipeline/pipeline.yaml`](pipeline/pipeline.yaml) into a Vertex AI
(Kubeflow) pipeline. The `condition:` on the register step becomes a `dsl.If` block, and the
compiled artifact carries a `comp-condition-1` component to prove it:

```
components: comp-condition-1, comp-deploy-staging, comp-evaluate,
            comp-ingest, comp-register, comp-train, comp-validate
steps behind the gate: deploy_staging, register
```

`deploy_staging` is pulled inside the branch too. A compiler that gated only the conditional
step would deploy a model it had just refused to register, and Kubeflow would not complain —
"deploy an unregistered model" is a perfectly legal DAG.

**Both branches are exercised.** The gate's floors are measured, not chosen for looking
reasonable:

- **0.83 absolute** — ranking the test split by a single raw sensor column scores 0.8257
  (`vibration_mm_s`). A model that cannot beat one column is worse than a spreadsheet, and
  everything downstream of it is pure cost.
- **+0.01 improvement** — test ROC AUC varies by sd 0.0087 across five seeds. A smaller gain
  is a different random split, not a better model.

```
candidate 0.8482 vs incumbent 0.8499 → GATE FAIL  improvement within noise    (exit 1)
candidate 0.8651 vs incumbent 0.8499 → GATE PASS  delta +0.0152               (exit 0)
```

## The portability verdict, in one line

The swap test passes on all three methods against a genuine second adapter. The write-up
argues that **the abstraction was not worth building for this system** — the seam did not
make migration cheap, because the expensive parts (IAM, pipelines, identity federation,
re-pricing; about 11 engineer-days, realistically four to five weeks) are not what it
abstracts. What it did do was force every provider detail into one directory, which is how
it became obvious that the *configuration* was never abstracted at all: `make
portability-audit` passes while `gs://` sits in `.dvc/config` and a GCP hostname sits in
`monitoring/dashboard.json`, inside a scanned directory, missed because the audit globs only
`*.py`.

## The least-privilege experiment that failed usefully

Removing the write scope from the credential and retrying an upload: **the write succeeded.**
The credential is an `authorized_user` whose `requires_scopes` is `False`, so the narrowed
scope list was discarded and the minted token still carried `cloud-platform`. Asking for a
smaller scope is not a boundary; the IAM role bound to the principal is. Every local run in
this project therefore executed with full project access, and the identity table is a design
rather than an enforced control until the three service accounts exist.

## The LLM half

**13 golden cases** in [`evals/golden/maintenance-triage.jsonl`](evals/golden/maintenance-triage.jsonl),
covering all four required categories: grounding (mt-004/005/006), guardrail (mt-007/008/009),
injection (mt-010/011) and insufficient input (mt-012/013).

```bash
make llm-eval-mine     # baseline: 13/13
make llm-gate-mine     # degraded prompt: GATE FAILED, 10 regressions, exits non-zero
make llm-cap-check     # 32-token output cap: GATE FAILED, 6 regressions
```

What survived the degraded run is as informative as what broke: the two easy decision
boundaries and the interlock refusal still passed, so a team watching only obvious capability
would have shipped a version that invents part numbers, leaks an operator's name, and obeys
instructions typed into a maintenance note.

**The token bill**, from provider usage fields at verified Vertex rates — 416 input and 37
output tokens per request:

| tier | model | THB per 1,000 requests |
|:--|:--|--:|
| small | gemini-2.5-flash-lite | 1.8559 |
| medium | gemini-2.5-flash | 7.1552 |
| large | gemini-2.5-pro | 29.2995 |

Capping `max_output_tokens` — the lever everyone reaches for first — **saves nothing here**,
because the answers already average 37 tokens; the 416-token instruction block is 90% of the
tokens and 57% of the cost. A 32-token cap saves 5.8% and fails the gate on 6 cases, because
the JSON stops mid-object. Caching the prefix would cut 41% and change no answers, with a
break-even hit rate of 21.7%.

---

## Reproduce it

```bash
make data
make pipeline          # compiles to reports/pipeline-vertex.yaml with the gate branch
python -m src.train --metrics-out reports/metrics.json
python scripts/evaluation_gate.py --metrics reports/metrics.json --incumbent 0.99   # FAILS, by design
make portability-audit
python scripts/portability_swap_check.py --second-provider local --endpoint http://127.0.0.1:8081
make llm-eval-mine && make llm-gate-mine
make cost EST=60 ACT=0.93 RPS=239.7 INSTANCE=e2-standard-4
```

## What is implemented but not yet run

The pipeline compiles but was not submitted; `generate()` reads Gemini's `usage_metadata`
but has not been called; the three service accounts do not exist yet. Every figure that
depends on those is labelled in the reports rather than blended with measured ones — the LLM
fixtures were authored to specify the contract, and the token counts behind the cost figures
come from the course's recorded fixture. `make llm-record` replaces them with real ones.
