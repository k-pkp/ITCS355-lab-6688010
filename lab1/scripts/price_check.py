"""Re-pull the cloud prices hard-coded in src/costs.py, straight from the provider.

    python scripts/price_check.py
    python scripts/price_check.py --region asia-southeast1 --fx 32.921586

Lab 2 asks you to verify the rates in src/costs.py rather than trust them, and Lab 5's
cost report is graded against real billing. Reading a pricing web page is verification you
cannot repeat six months later; reading the Cloud Billing Catalog API is the same number,
dated, and reproducible — the console pricing page renders these very SKUs.

Composite machine types have no single SKU. They are billed per vCPU and per GiB of RAM,
so this script rebuilds each machine type from its component SKUs and shows the working.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, costs

# Cloud Billing Catalog service identifiers. Stable, and printed by --list-services.
COMPUTE_ENGINE_SERVICE = "6F81-5844-456A"
VERTEX_AI_SERVICE = "C7E2-9256-1C43"
CLOUD_RUN_SERVICE = "152E-C115-5142"

# Each machine type as its billable parts: (service, SKU description prefix, quantity).
# Quantities are vCPU counts and GiB of RAM for the machine shape being priced.
MACHINE_RECIPES: dict[str, list[tuple[str, str, float]]] = {
    "e2-standard-4": [
        (COMPUTE_ENGINE_SERVICE, "E2 Instance Core running in", 4),
        (COMPUTE_ENGINE_SERVICE, "E2 Instance Ram running in", 16),
    ],
    "n1-standard-4": [
        (COMPUTE_ENGINE_SERVICE, "N1 Predefined Instance Core running in", 4),
        (COMPUTE_ENGINE_SERVICE, "N1 Predefined Instance Ram running in", 15),
    ],
    "n1-standard-4+t4": [
        (COMPUTE_ENGINE_SERVICE, "N1 Predefined Instance Core running in", 4),
        (COMPUTE_ENGINE_SERVICE, "N1 Predefined Instance Ram running in", 15),
        (COMPUTE_ENGINE_SERVICE, "Nvidia Tesla T4 GPU running in", 1),
    ],
    "vertex-training-e2-standard-4": [
        (VERTEX_AI_SERVICE, "Vertex AI: Training/Pipelines on E2 Instance Core running in", 4),
        (VERTEX_AI_SERVICE, "Vertex AI: Training/Pipelines on E2 Instance Ram running in", 16),
        (VERTEX_AI_SERVICE, "Vertex AI: Training/Pipelines Management fee on E2 Instance Core in", 4),
        (VERTEX_AI_SERVICE, "Vertex AI: Training/Pipelines Management fee on E2 Instance RAM in", 16),
    ],
    "vertex-training-n1-standard-4": [
        (VERTEX_AI_SERVICE, "Vertex AI: Training/Pipelines on N1 Predefined Instance Core running in", 4),
        (VERTEX_AI_SERVICE, "Vertex AI: Training/Pipelines on N1 Predefined Instance Ram running in", 15),
        (VERTEX_AI_SERVICE, "Vertex AI: Training/Pipelines Management fee on N1 Instance Core in", 4),
        (VERTEX_AI_SERVICE, "Vertex AI: Training/Pipelines Management fee on N1 Instance RAM in", 15),
    ],
    "vertex-endpoint-n1-standard-4": [
        (VERTEX_AI_SERVICE, "Vertex AI: Online/Batch Prediction N1 Predefined Instance Core running in", 4),
        (VERTEX_AI_SERVICE, "Vertex AI: Online/Batch Prediction N1 Predefined Instance Ram running in", 15),
    ],
    # Cloud Run bills per second; 3600 turns one vCPU-second into one vCPU-hour.
    "cloud-run-1vcpu-2gib": [
        (CLOUD_RUN_SERVICE, "Services CPU (Instance-based billing) in", 3600),
        (CLOUD_RUN_SERVICE, "Services Memory (Instance-based billing) in", 7200),
    ],
}

# Spot equivalents, so the discount factor is measured rather than assumed.
SPOT_RECIPES: dict[str, list[tuple[str, str, float]]] = {
    "e2-standard-4": [
        (COMPUTE_ENGINE_SERVICE, "Spot Preemptible E2 Instance Core running in", 4),
        (COMPUTE_ENGINE_SERVICE, "Spot Preemptible E2 Instance Ram running in", 16),
    ],
    "n1-standard-4": [
        (COMPUTE_ENGINE_SERVICE, "Spot Preemptible N1 Predefined Instance Core running in", 4),
        (COMPUTE_ENGINE_SERVICE, "Spot Preemptible N1 Predefined Instance Ram running in", 15),
    ],
    "n1-standard-4+t4": [
        (COMPUTE_ENGINE_SERVICE, "Spot Preemptible N1 Predefined Instance Core running in", 4),
        (COMPUTE_ENGINE_SERVICE, "Spot Preemptible N1 Predefined Instance Ram running in", 15),
        (COMPUTE_ENGINE_SERVICE, "Nvidia Tesla T4 GPU attached to Spot Preemptible VMs running in", 1),
    ],
}


def access_token() -> str:
    """Mint an OAuth token from the ambient GCP credentials.

    Layer 3 territory: this script is a provider tool, not part of src/.
    """
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def fetch_json(url: str, token: str, attempts: int = 5) -> dict:
    """GET one JSON page, retrying the catalog API's intermittent 502s."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(3)
    raise RuntimeError("unreachable")


def load_sku_prices(service_id: str, region: str, token: str) -> dict[str, float]:
    """Return {sku description: USD unit price} for one service in one region."""
    prices: dict[str, float] = {}
    page_token = ""
    while True:
        url = (
            f"https://cloudbilling.googleapis.com/v1/services/{service_id}"
            f"/skus?pageSize=2000&currencyCode=USD"
        )
        if page_token:
            url += "&pageToken=" + page_token

        page = fetch_json(url, token)
        for sku in page.get("skus", []):
            if region not in sku.get("serviceRegions", []):
                continue
            for pricing_info in sku.get("pricingInfo", []):
                for tier in pricing_info["pricingExpression"]["tieredRates"]:
                    units = int(tier["unitPrice"].get("units", 0))
                    nanos = int(tier["unitPrice"].get("nanos", 0))
                    unit_price = units + nanos / 1e9
                    if unit_price > 0:
                        prices[sku["description"]] = unit_price

        page_token = page.get("nextPageToken") or ""
        if not page_token:
            return prices


def find_price(prices: dict[str, float], description_prefix: str) -> float:
    """Find the one SKU whose description starts with `description_prefix`.

    Matching on a prefix rather than anywhere in the string matters: "E2 Instance Core
    running in Singapore" is a substring of "Spot Preemptible E2 Instance Core running
    in Singapore", and silently pricing on-demand compute at the spot rate is the exact
    class of error this script exists to catch.
    """
    matches = []
    for description in prices:
        if description.lower().startswith(description_prefix.lower()):
            matches.append(description)

    if not matches:
        raise KeyError(f"no SKU starting with {description_prefix!r} in this region")
    if len(matches) > 1:
        raise KeyError(f"{description_prefix!r} matches {len(matches)} SKUs: {matches[:4]}")
    return prices[matches[0]]


def price_recipe(
    recipe: list[tuple[str, str, float]],
    price_books: dict[str, dict[str, float]],
) -> tuple[float, list[str]]:
    """Sum a machine recipe into USD per hour, returning the total and the working."""
    total_usd = 0.0
    working: list[str] = []
    for service_id, description_prefix, quantity in recipe:
        unit_price = find_price(price_books[service_id], description_prefix)
        line_total = unit_price * quantity
        total_usd += line_total
        working.append(
            f"{quantity:g} x {unit_price:.9f} = {line_total:.6f}  ({description_prefix})"
        )
    return total_usd, working


def parse_command_line() -> argparse.Namespace:
    """Command line for the price check."""
    parser = argparse.ArgumentParser(description="Verify src/costs.py against live SKU prices")
    parser.add_argument("--region", default=None, help="defaults to REGION from cloud.env")
    parser.add_argument("--fx", type=float, default=32.921586,
                        help="THB per USD; the rate quoted in src/costs.py")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="allowed fractional drift before a row is reported STALE")
    parser.add_argument("--show-working", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Compare every GCP rate in src/costs.py against the live catalog."""
    options = parse_command_line()
    loaded_config = config.load(strict=False)
    region = options.region or loaded_config.region

    token = access_token()
    price_books: dict[str, dict[str, float]] = {}
    for service_id in (COMPUTE_ENGINE_SERVICE, VERTEX_AI_SERVICE, CLOUD_RUN_SERVICE):
        price_books[service_id] = load_sku_prices(service_id, region, token)

    print(f"Cloud Billing Catalog · region {region} · 1 USD = {options.fx} THB\n")
    print(f"  {'instance':32} {'live THB/h':>11} {'in costs.py':>12}  status")

    stale_rows = 0
    for instance, recipe in MACHINE_RECIPES.items():
        live_usd, working = price_recipe(recipe, price_books)
        live_thb = live_usd * options.fx
        recorded_thb = costs.PRICE_TABLE["gcp"][instance]

        drift = abs(live_thb - recorded_thb) / recorded_thb
        status = "ok" if drift <= options.tolerance else f"STALE ({drift * 100:.1f}% drift)"
        if drift > options.tolerance:
            stale_rows += 1

        print(f"  {instance:32} {live_thb:>11.3f} {recorded_thb:>12.3f}  {status}")
        if options.show_working:
            for entry in working:
                print(f"      {entry}")

    print("\n  spot discount factors (measured, not assumed)")
    for instance, spot_recipe in SPOT_RECIPES.items():
        on_demand_usd, _ = price_recipe(MACHINE_RECIPES[instance], price_books)
        spot_usd, _ = price_recipe(spot_recipe, price_books)
        live_factor = spot_usd / on_demand_usd
        recorded_factor = costs.SPOT_FACTOR_BY_INSTANCE[instance]
        print(f"  {instance:32} {live_factor:>11.4f} {recorded_factor:>12.4f}")

    if stale_rows:
        print(f"\n{stale_rows} rate(s) drifted beyond {options.tolerance * 100:.0f}%. "
              "Update src/costs.py and re-date the comment above PRICE_TABLE.")
        return 1

    print("\nAll recorded GCP rates match the live catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
