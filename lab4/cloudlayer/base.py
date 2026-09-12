"""The portability seam.

Eleven methods. Every managed ML platform sells you the same eleven operations under
different names; writing this once is the difference between knowing a product and
knowing the category. `generate` is the newest of them and the one the vendors are
currently busiest renaming.

You implement exactly ONE of aws.py, azure.py, or gcp.py. Lab 1 needs only `upload`,
`download`, and `push_image`. The rest raise NotImplementedError until the lab that
needs them, which is deliberate — do not implement ahead.

Nothing outside cloudlayer/ may import a provider SDK.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CloudAdapter(ABC):
    """Provider-neutral interface. The grader only ever calls these."""

    def __init__(self, config) -> None:
        """Hold the resolved configuration every adapter method reads its endpoints from."""
        self.config = config

    # --- Lab 1 ---------------------------------------------------------------
    @abstractmethod
    def upload(self, local_path: str, key: str) -> str:
        """Upload a file under BLOB_URI. Returns the full URI of the stored object."""

    @abstractmethod
    def download(self, uri: str, local_path: str) -> None:
        """Fetch an object to a local path. Creates parent directories."""

    @abstractmethod
    def push_image(self, local_tag: str) -> str:
        """Push a locally built image to CONTAINER_REGISTRY. Returns the remote reference,
        which must be digest-pinned (repo@sha256:...), not tag-pinned."""

    # --- Lab 2 ---------------------------------------------------------------
    def submit_training(self, image_uri: str, arguments: dict[str, Any]) -> str:
        """Start a managed training job from an already-pushed image. Returns a job id."""
        raise NotImplementedError("Lab 2")

    def wait_training(self, job_id: str) -> dict[str, Any]:
        """Block until a submitted job reaches a terminal state. Returns its final status."""
        raise NotImplementedError("Lab 2")

    def register_model(self, model_uri: str, name: str) -> str:
        """Put a trained artifact in the model registry. Returns the version string."""
        raise NotImplementedError("Lab 2")

    # --- Lab 3 ---------------------------------------------------------------
    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        """Put a registered version behind an endpoint. Returns the endpoint reference."""
        raise NotImplementedError("Lab 3")

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a deployed endpoint once, normalising the provider's response shape."""
        raise NotImplementedError("Lab 3")

    # --- Lab 4 ---------------------------------------------------------------
    def emit_metric(self, name: str, value: float, unit: str = "None") -> None:
        """Write one custom metric point to the provider's monitoring service."""
        raise NotImplementedError("Lab 4")

    # --- Lab 5 ---------------------------------------------------------------
    def generate(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a managed LLM endpoint once. Returns at least:

            {"response": str, "input_tokens": int,
             "output_tokens": int, "latency_ms": int}

        Token counts must be read from the provider's own usage fields. Estimating them
        by counting words produces a cost report built on a number you made up, and
        every provider tokenises differently.

        `params` carries the knobs worth exposing at the seam — `max_output_tokens`
        above all, since output tokens dominate the bill. Provider-specific options
        belong here in Layer 3, never in the caller.
        """
        raise NotImplementedError("Lab 5")

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Delete every resource carrying these tags. Returns what was deleted.

        Deletion is asynchronous on all three providers — returning successfully does
        not mean the resource is gone. Re-check, and check the bill.
        """
        raise NotImplementedError("Lab 5")


class LocalAdapter(CloudAdapter):
    """Filesystem stand-in so Lab 1 runs before your cloud account is ready.

    Acceptable for developing Lab 1. NOT acceptable for submission — your submitted
    Lab 1 must upload to real object storage and push to a real registry.
    """

    def upload(self, local_path: str, key: str) -> str:
        """Copy a file into the filesystem stand-in for object storage.

        Returns a local:// URI, which download() below is the only thing that understands.
        Callers pass that handle straight back rather than rebuilding it from a bucket
        name and a key, because only this adapter knows the grammar.
        """
        import shutil
        from pathlib import Path

        destination = Path(self.config.data_dir) / "_local_blob" / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)
        return f"local://{destination}"

    def download(self, uri: str, local_path: str) -> None:
        """Copy a file back out of the filesystem stand-in, creating parent directories."""
        import shutil
        from pathlib import Path

        source_path = uri.removeprefix("local://")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, local_path)

    def push_image(self, local_tag: str) -> str:
        """Refuse: there is no local registry, and pretending otherwise would hide it."""
        raise NotImplementedError(
            "LocalAdapter cannot push images. Implement your provider's adapter for Lab 1 submission."
        )

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a service running on this machine over plain HTTP.

        Present so that Lab 5's swap test can exercise a genuine second implementation of
        upload, download and invoke without a second cloud account and a second bill. The
        service it calls is the same image that would be deployed to a managed endpoint,
        so what is being proved is real: the caller's code does not change, only the
        adapter behind it.

        `endpoint` is a base URL such as http://127.0.0.1:8080.
        """
        import json
        import urllib.request

        request = urllib.request.Request(
            f"{endpoint.rstrip('/')}/predict",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
