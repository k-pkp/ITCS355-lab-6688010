"""Lab 5 Task 4 — prove the portability seam is real.

    python scripts/portability_swap_check.py --second-provider gcp

Exercises upload, download, and invoke against a SECOND provider's adapter. You are not
migrating the whole system; you are proving the seam exists.

If this fails while `make portability-audit` passes, you have found something worth
writing about: the leak was in configuration or in an assumption, not in an import.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config


def main() -> int:
    """Exercise upload, download and invoke through a second provider's adapter."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--second-provider", required=True,
                    choices=["aws", "azure", "gcp", "local"])
    argument_parser.add_argument("--endpoint", default="http://127.0.0.1:8080",
                    help="endpoint the second adapter's invoke() should call")
    options = argument_parser.parse_args()

    primary = config.load(strict=False)
    if primary.provider == options.second_provider:
        print("The second provider must differ from CLOUD_PROVIDER.")
        return 1

    print(f"primary   {primary.provider}")
    print(f"secondary {options.second_provider}\n")

    # Same Config object, different adapter. If your code needs more than this to switch,
    # say so in the write-up — that IS the finding.
    swapped = config.Config(**{**primary.__dict__, "provider": options.second_provider})
    adapter = get_adapter(swapped)

    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as temporary_directory:
        probe = Path(temporary_directory) / "probe.txt"
        probe.write_text("itcs355 portability probe\n")

        # download() is handed the URI that upload() returned, not a URI this script built
        # by concatenating BLOB_URI with a key. Building it here would embed one provider's
        # URI grammar in provider-neutral code — gs:// and s3:// happen to share a shape,
        # local:// does not, and Azure's does not either. Passing the returned handle
        # around is the portable habit, and this script has to follow its own rule.
        uploaded: dict[str, str] = {}

        def do_upload() -> None:
            """Upload the probe file and remember the URI the adapter gave back."""
            uploaded["uri"] = adapter.upload(str(probe), "portability/probe.txt")

        def do_download() -> None:
            """Fetch the probe back using the adapter's own URI, and check the bytes."""
            destination = Path(temporary_directory) / "back.txt"
            adapter.download(uploaded["uri"], str(destination))
            if destination.read_text() != probe.read_text():
                raise ValueError("round-tripped file does not match what was uploaded")

        for name, call in (
            ("upload", do_upload),
            ("download", do_download),
            # A real payload, not an empty one: an invoke that returns 422 has proved the
            # call reached something, but not that the seam carries a prediction back.
            ("invoke", lambda: adapter.invoke(options.endpoint, {
                "temp_c": 78.4, "vibration_mm_s": 3.1, "pressure_kpa": 315.2,
                "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0,
            })),
        ):
            try:
                call()
                results[name] = "PASS"
            except NotImplementedError:
                results[name] = "NOT IMPLEMENTED"
            except Exception as exc:
                results[name] = f"FAIL — {type(exc).__name__}: {exc}"

    for name, outcome in results.items():
        print(f"  [{outcome.split(' —')[0]:<16}] {name}  {outcome.partition('— ')[2]}")

    print("\nWrite-up (Lab 5 Task 4), half a page:")
    print("  - which method was hardest to port, and why")
    print("  - one place the abstraction genuinely leaked and could not be hidden")
    print("  - what a full migration would cost in engineering days")
    print("  - was building this abstraction worth it? A well-argued 'no' scores full marks.")

    return 0 if all(outcome == "PASS" for outcome in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
