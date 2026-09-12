"""Lab 4 — unit tests for the drift statistics.

These are the "unit test" category: no network, no model, no data file. They pin the
behaviour of the two statistics the alerting threshold is expressed in, so that a change to
the maths cannot silently change what the threshold means.
"""
from __future__ import annotations

import numpy as np

from monitoring import drift


def test_psi_is_zero_for_identical_samples():
    """Comparing a sample against itself must score no drift at all."""
    values = np.random.default_rng(1).normal(size=2000)
    assert drift.psi(values, values) < 1e-9


def test_psi_grows_with_the_size_of_the_shift():
    """A bigger shift must score higher, or the threshold means nothing."""
    generator = np.random.default_rng(2)
    reference = generator.normal(size=4000)
    small_shift = drift.psi(reference, reference + 0.25)
    large_shift = drift.psi(reference, reference + 1.5)
    assert small_shift < large_shift


def test_psi_sees_a_spread_change_that_leaves_the_mean_alone():
    """The failure a mean-based monitor misses, and the reason PSI is on the dashboard."""
    reference = np.random.default_rng(3).normal(size=4000)
    widened = reference * 1.6
    assert abs(float(widened.mean()) - float(reference.mean())) < 0.05
    assert drift.psi(reference, widened) > drift.PSI_ALERT


def test_psi_survives_an_empty_bin():
    """Laplace smoothing must keep an unpopulated bin from producing an infinite score."""
    reference = np.linspace(0, 10, 1000)
    current = np.linspace(0, 1, 1000)
    score = drift.psi(reference, current)
    assert np.isfinite(score)


def test_ks_statistic_is_bounded_and_zero_for_identical_samples():
    """KS is a maximum CDF gap, so it lives in [0, 1] and is 0 against itself."""
    values = np.random.default_rng(4).normal(size=1000)
    assert drift.ks_statistic(values, values) == 0.0
    shifted = drift.ks_statistic(values, values + 5)
    assert 0.0 < shifted <= 1.0


def test_ks_reacts_more_to_location_than_psi_does_to_spread():
    """The two statistics are complementary; this pins which is which.

    KS is the location detector: a pure translation moves it hard. PSI is the shape
    detector: it is what catches the widening above, which KS barely registers.
    """
    reference = np.random.default_rng(5).normal(size=4000)
    translated = reference + 1.0
    widened = reference * 1.6

    assert drift.ks_statistic(reference, translated) > drift.ks_statistic(reference, widened)


def test_verdict_bands_match_the_documented_constants():
    """The reported verdict must follow the constants, not a hard-coded number."""
    assert drift.verdict_for(drift.PSI_NO_CHANGE - 0.001) == "stable"
    assert drift.verdict_for(drift.PSI_NO_CHANGE + 0.001) == "moderate"
    assert drift.verdict_for(drift.PSI_MODERATE + 0.001) == "significant"


def test_alert_threshold_sits_above_the_measured_noise_floor():
    """Guards the justification in monitoring/drift.py.

    The noise study measured a maximum PSI of 0.0432 between the reference and a 600-row
    window of unchanged data. If somebody lowers PSI_ALERT below that, this fails and says
    why, rather than the alert quietly firing every week on nothing.
    """
    measured_noise_ceiling = 0.0432
    assert drift.PSI_ALERT > 2 * measured_noise_ceiling
