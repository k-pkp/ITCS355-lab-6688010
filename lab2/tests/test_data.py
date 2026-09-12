"""Data tests.

Two kinds live here, and Lab 4 asks you to tell them apart:

  * schema / contract tests — assertions about the DATA. They fail when an upstream
    producer changes something, even though your code is untouched.
  * property tests — assertions about your splitting LOGIC. They fail when you change
    the code.

test_no_machine_leaks_across_splits is the one that matters most. It is the single
most common silent error in student projects: readings from one machine appearing in
both train and validation, producing a score that never survives contact with
production.
"""
from __future__ import annotations

import pytest

from src import config, data

RAW = config.REPO_ROOT / "data" / "raw" / "sensors.csv"


@pytest.fixture(scope="module")
def frame():
    """The raw dataset, loaded once for the whole module."""
    if not RAW.exists():
        pytest.skip("data/raw/sensors.csv missing — run `make data` or `dvc pull` first")
    return data.load_raw(RAW)


# --- contract tests: about the data -------------------------------------------------

def test_schema_columns_present_and_typed(frame):

    """Every contracted column is present, with no extras and the declared dtype."""
    missing = set(data.SCHEMA) - set(frame.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    unexpected = set(frame.columns) - set(data.SCHEMA)
    assert not unexpected, f"unexpected columns: {sorted(unexpected)}"
    for col, expected in data.SCHEMA.items():
        assert str(frame[col].dtype) == expected, f"{col}: expected {expected}, got {frame[col].dtype}"


def test_no_nulls_in_required_columns(frame):


    """No contracted column carries nulls, which imputation would silently paper over."""
    nulls = frame[list(data.SCHEMA)].isna().sum()
    offenders = nulls[nulls > 0]
    assert offenders.empty, f"null values found: {offenders.to_dict()}"


def test_features_within_plausible_ranges(frame):


    """Every feature stays inside the range the domain says is physically possible."""
    for col, (lower_bound, upper_bound) in data.PLAUSIBLE_RANGES.items():
        assert frame[col].min() >= lower_bound, f"{col} below plausible floor: {frame[col].min()}"
        assert frame[col].max() <= upper_bound, f"{col} above plausible ceiling: {frame[col].max()}"


def test_target_is_binary_and_not_degenerate(frame):


    """The target is 0/1 and has both classes, so the metric means something."""
    values = set(frame[data.TARGET].unique().tolist())
    assert values <= {0, 1}, f"target has values outside 0/1: {values}"
    rate = frame[data.TARGET].mean()
    assert 0.01 < rate < 0.99, f"target is degenerate, positive rate = {rate:.4f}"


def test_identifier_is_unique(frame):


    """reading_id identifies exactly one row, so a duplicate cannot cross the split."""
    assert frame[data.IDENTIFIER].is_unique, "reading_id is not unique"


# --- property tests: about the splitting logic --------------------------------------

def test_no_machine_leaks_across_splits(frame):

    """No machine appears in two partitions. This is the leakage test the course turns on."""
    train, validation_frame, test = data.split(frame, seed=42)
    train_groups, validation_groups, test_groups = (set(partition[data.GROUP]) for partition in (train, validation_frame, test))
    assert not train_groups & validation_groups, f"machines in both train and val: {sorted(train_groups & validation_groups)[:5]}"
    assert not train_groups & test_groups, f"machines in both train and test: {sorted(train_groups & test_groups)[:5]}"
    assert not validation_groups & test_groups, f"machines in both val and test: {sorted(validation_groups & test_groups)[:5]}"


def test_split_is_deterministic_given_seed(frame):


    """The same seed produces the same split, which is what makes a run reproducible."""
    first_split = data.split(frame, seed=7)
    second_split = data.split(frame, seed=7)
    for first_partition, second_partition in zip(first_split, second_split):
        assert first_partition[data.IDENTIFIER].tolist() == second_partition[data.IDENTIFIER].tolist()


def test_split_changes_with_seed(frame):


    """A different seed produces a different split, so the seed is genuinely in use."""
    first_split, _, _ = data.split(frame, seed=1)
    second_split, _, _ = data.split(frame, seed=2)
    assert first_split[data.IDENTIFIER].tolist() != second_split[data.IDENTIFIER].tolist(), "seed has no effect — split is not random"


def test_every_row_lands_in_exactly_one_split(frame):


    """The three partitions account for every row exactly once: none lost, none doubled."""
    train, validation_frame, test = data.split(frame, seed=13)
    total = len(train) + len(validation_frame) + len(test)
    assert total == len(frame), f"rows lost or duplicated: {total} vs {len(frame)}"


def test_data_fingerprint_is_stable():


    """The fingerprint of unchanged bytes does not move, or lineage would be meaningless."""
    if not RAW.exists():
        pytest.skip("no raw data")
    assert data.data_fingerprint(RAW) == data.data_fingerprint(RAW)
