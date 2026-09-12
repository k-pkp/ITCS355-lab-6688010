"""Cost model. Used by Labs 2, 3, and 5.

Prices change and differ by region, so these are STARTING VALUES you must verify against
your provider's calculator. Verifying them is part of Lab 2; citing a stale number without
checking is the kind of thing the cost report is designed to catch.

Rates are THB per hour, on-demand. Discounted (spot / low-priority / preemptible) compute
is cheaper by a factor that varies by machine family and region — SPOT_FACTOR_BY_INSTANCE
below carries the measured factors, and the single global constant is only a fallback.
"""
from __future__ import annotations

# VERIFIED 2026-09-07 against the Cloud Billing Catalog API — the same SKU prices the
# console pricing page renders — for region asia-southeast1 (Singapore), the region in
# cloud.env. Converted at 1 USD = 32.921586 THB (open.er-api.com, 2026-09-07 00:02 UTC).
# Reproduce the pull with scripts/price_check.py; the workings are in
# reports/lab2-comparison.md.
#
# Composite machine types are priced per vCPU and per GiB of RAM, so the hourly figures
# below are sums, not published line items:
#   e2-standard-4  = 4 x 0.026909310 + 16 x 0.003605940 USD/h = 0.165332 USD/h
#   n1-standard-4  = 4 x 0.038999    + 15 x 0.005226    USD/h = 0.234386 USD/h
#
# The aws/ and azure/ rates below are the course's starting values and have NOT been
# re-verified — this project runs on GCP, and quoting an unchecked rate for a provider
# you never priced is exactly the fabrication the cost report is meant to catch.
PRICE_TABLE: dict[str, dict[str, float]] = {
    "local": {"local": 0.0},
    "aws": {
        "ml.m5.large": 4.2,
        "ml.m5.xlarge": 8.4,
        "ml.c5.xlarge": 7.3,
        "ml.g4dn.xlarge": 26.0,
    },
    "azure": {
        "Standard_DS3_v2": 8.1,
        "Standard_F4s_v2": 6.9,
        "Standard_NC4as_T4_v3": 24.5,
    },
    "gcp": {
        # Raw Compute Engine VMs.
        "e2-standard-4": 5.443,
        "n1-standard-4": 7.716,
        "n1-standard-4+t4": 19.897,
        # Vertex AI custom training. The managed premium over the raw VM is real and
        # visible: 7.076 against 5.443 for the same four cores, a 30% management fee.
        "vertex-training-e2-standard-4": 7.076,
        "vertex-training-n1-standard-4": 10.031,
        # Vertex AI online prediction, billed for every hour the endpoint exists.
        "vertex-endpoint-n1-standard-4": 8.889,
        # Cloud Run, 1 vCPU and 2 GiB, instance-based billing. Scales to zero, which is
        # why Lab 3 and Lab 5 compare it against the always-on endpoint above.
        "cloud-run-1vcpu-2gib": 3.129,
    },
}

# Measured, not assumed. The course scaffold said discounted compute is "roughly 30% of
# on-demand across all three providers"; on GCP in Singapore that is true of N1 (0.268)
# and badly wrong for E2 (0.546). Spot discounts are set per machine family and per
# region, so a single global factor quietly doubles or halves your estimate.
SPOT_FACTOR_BY_INSTANCE: dict[str, float] = {
    "e2-standard-4": 0.5455,
    "vertex-training-e2-standard-4": 0.5455,
    "n1-standard-4": 0.2678,
    "vertex-training-n1-standard-4": 0.2678,
    "n1-standard-4+t4": 0.2809,
}

# Fallback for any instance with no measured factor, including the unverified aws and
# azure rows. It is the course's original assumption, kept only as a last resort.
SPOT_FACTOR = 0.30

DEFAULT_UTILISATIONS = (0.05, 0.25, 0.80)


def hourly_rate(provider: str, instance: str, spot: bool = False) -> float:
    """Return the THB-per-hour rate for one instance type on one provider.

    Raises rather than guessing when the instance is unknown: substituting a similar
    machine type is how a cost report ends up describing a system nobody ran.
    """
    table = PRICE_TABLE.get(provider.lower())
    if table is None:
        raise KeyError(f"No price table for provider {provider!r}. Add it to src/costs.py.")
    if instance not in table:
        raise KeyError(
            f"No rate for {instance!r} on {provider}. Known: {sorted(table)}. "
            "Add the instance you actually used — do not substitute a similar one silently."
        )
    on_demand = table[instance]
    if not spot:
        return on_demand

    discount = SPOT_FACTOR_BY_INSTANCE.get(instance, SPOT_FACTOR)
    return on_demand * discount


def cost_per_1k_predictions(
    hourly_thb: float,
    throughput_rps: float,
    utilisation: float,
) -> float:
    """Cost of 1,000 predictions on an always-on endpoint.

    utilisation is the fraction of provisioned capacity you actually use. It is the most
    fragile number in any serving cost estimate, which is why Lab 3 makes you state it
    explicitly and Lab 5 makes you report three of them.
    """
    if not 0 < utilisation <= 1:
        raise ValueError("utilisation must be in (0, 1]")
    if throughput_rps <= 0:
        raise ValueError("throughput_rps must be positive")
    effective_rps = throughput_rps * utilisation
    seconds_per_1k = 1000.0 / effective_rps
    return hourly_thb * (seconds_per_1k / 3600.0)


def batch_breakeven_rps(
    endpoint_hourly_thb: float,
    batch_job_thb: float,
    batch_runs_per_day: int = 1,
) -> float:
    """Request rate below which scheduled batch inference is cheaper than a warm endpoint.

    Lab 3 asks you to compute this for your own service. The answer is usually lower than
    students expect, which is the point.
    """
    endpoint_daily = endpoint_hourly_thb * 24
    batch_daily = batch_job_thb * batch_runs_per_day
    if batch_daily >= endpoint_daily:
        return 0.0
    return (endpoint_daily - batch_daily) / 86400.0
