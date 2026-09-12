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

The steps below are the CI `test` job from `.github/workflows/ci.yml`, in its order. Cheap
checks first, so a schema mistake fails in seconds rather than after an image build.

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

It does not prove anything about GitHub's runners: the steps above were executed locally,
command for command, from the workflow file. The repository has no Actions history yet, so
the run to attach to a pull request comes with the first push. The pull request itself stays
open and unmerged, which is the deliverable — `demo/break-data-contract` is pushed and must
not be merged.

## Restoring

`main` is unaffected; the break lives only on the branch. After checking `main` out again,
`make data` regenerates the six-column dataset and `tests/test_data.py` passes 10/10.
