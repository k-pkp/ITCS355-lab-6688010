# Lab 4 Task 3 — a bad commit, blocked

A green pipeline proves nothing about whether the tests work. This is the failing run.

## The change

Branch `demo/break-data-contract`, commit `bde3614`. One line added to
`scripts/make_dataset.py`:

```python
df = df.drop(columns=["ambient_humidity"])
```

Chosen because it is the shape a real bad commit takes. `ambient_humidity` carries 4.9% of
the model's importance and reads noisy, so dropping it from the export looks like tidying.
The generator still runs, the CSV still loads, pandas raises nothing, and the model still
trains — on five features instead of six, quietly.

## The run

**Pull request [#1](https://github.com/k-pkp/ITCS355-lab-6688010/pull/1)** ·
**[failing run](https://github.com/k-pkp/ITCS355-lab-6688010/actions/runs/34689099140/job/103541119853)**
· `CI / test` failed after 47s · opened and closed without merging.

| Step | Result | Duration |
|:--|:--|--:|
| Set up job | pass | 0s |
| `actions/checkout@v4` | pass | 1s |
| `actions/setup-python@v5` | pass | 4s |
| Install (`pip install --require-hashes`) | pass | 35s |
| Lint | pass | 0s |
| Portability audit | pass | 0s |
| Generate dataset | pass | 1s |
| Unit tests | pass | 1s |
| **Data contract tests** | **FAIL** | 1s |
| Model behaviour tests | not run | — |
| Service tests | not run | — |
| `build` job (image + integration test) | **skipped** | — |

Two things in that table are the point of the task.

**The failure arrived about 42 seconds in, and one second into the step that found it.**
Everything before it — lint, the portability audit, dataset generation, the unit tests —
takes a second or less, because the sequence is ordered cheapest-first on purpose. A schema
mistake does not wait for an image build to be told it is a schema mistake.

**The `build` job never started.** It declares `needs: test`, so no image was built, nothing
was pushed, and CD — which triggers on a *successful* CI run — never fired. The bad commit
did not reach anything.

The install step passing is worth noting too: 35 seconds of
`pip install --require-hashes -r requirements.txt` on a runner that is not the author's
machine is the lock file from Lab 1 doing its job under the strict flag.

Local reproduction of the same failure, command for command from the workflow file:

```
=== Lint ===
All checks passed!

=== Portability audit ===
PORTABILITY AUDIT PASSED — src, service, monitoring, tests contain no provider-specific strings

=== Unit tests ===
8 passed in 0.29s

=== Data contract tests ===
FAILED tests/test_data.py::test_schema_columns_present_and_typed - AssertionError
FAILED tests/test_data.py::test_no_nulls_in_required_columns - KeyError
FAILED tests/test_data.py::test_features_within_plausible_ranges - KeyError
3 failed, 7 passed in 0.59s
```

## The test that caught it, and the message

`tests/test_data.py::test_schema_columns_present_and_typed`:

```
E       AssertionError: missing columns: ['ambient_humidity']
E       assert not {'ambient_humidity'}

tests/test_data.py:35: AssertionError
```

Three tests failed, and the order they failed in is the useful part. The schema test names
the problem — a column is missing. The other two fail with a bare `KeyError` because they
try to look up a column that is not there; they are collateral, not diagnosis. A suite where
only the `KeyError`s fired would tell you something broke without telling you what.

Nothing after this point in the pipeline runs: the `build` job needs `test`, so no image is
built, no image is pushed, and CD never triggers because it waits on a successful CI run.

## What this proves, and what it does not

It proves the contract tests fail for the reason they were written for, on a change that
looks reasonable in review. A reviewer skimming a one-line diff that deletes a low-importance
column would likely approve it.

It proves it on GitHub's runners as well as locally: the run linked above is a real
`pull_request` trigger on a clean checkout, not a rehearsal on the author's machine.

The pull request was closed without merging, which is the deliverable. The branch
`demo/break-data-contract` stays on the remote so the diff and the failing run remain
reachable.

## Restoring

`main` is unaffected; the break lives only on the branch. After checking `main` out again,
`make data` regenerates the six-column dataset and `tests/test_data.py` passes 10/10.
