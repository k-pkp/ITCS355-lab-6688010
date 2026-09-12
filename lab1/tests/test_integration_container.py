"""Lab 4 — integration test: the real serving container, over real HTTP.

Distinct from tests/test_service.py, which drives the FastAPI app in-process. That one
proves the routes behave; this one proves the *image* does — that the dependency layer was
copied into the runtime stage, that the non-root user can read what it needs, and that the
declared port actually serves. Those are exactly the failures an in-process test cannot
see, and they are the ones that break a deploy.

Skipped when Docker is unavailable, so the suite still runs on a machine without it. CI has
Docker, and CI is where this test earns its place.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import pytest

from src import config

IMAGE = "itcs355-serve:test"
CONTAINER_PORT = 8099
VALID_ROW = {
    "temp_c": 78.4, "vibration_mm_s": 3.1, "pressure_kpa": 315.2,
    "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0,
}


def docker_is_available() -> bool:
    """True when a Docker daemon is reachable, not merely when the client is installed."""
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True)
    return probe.returncode == 0


def http_post(url: str, body: dict) -> tuple[int, dict]:
    """POST JSON and return the status code and parsed body, 4xx and 5xx included."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


@pytest.fixture(scope="module")
def running_container():
    """Build the serving image, start it, wait for readiness, and always clean up."""
    if not docker_is_available():
        pytest.skip("no Docker daemon available")

    repository_root = config.REPO_ROOT
    model_path = repository_root / "reports" / "model.joblib"
    if not model_path.exists():
        subprocess.run(
            ["python", "scripts/export_model.py", "--out", str(model_path)],
            cwd=repository_root, check=True, capture_output=True,
        )

    subprocess.run(
        ["docker", "build", "-f", "service/Dockerfile.serve", "-t", IMAGE, "."],
        cwd=repository_root, check=True, capture_output=True,
    )

    container_name = f"itcs355-itest-{uuid.uuid4().hex[:8]}"
    subprocess.run([
        "docker", "run", "-d", "--rm", "--name", container_name,
        "-p", f"{CONTAINER_PORT}:8080",
        "-v", f"{repository_root / 'reports'}:/app/reports:ro",
        "-e", "MODEL_PATH=/app/reports/model.joblib",
        "-e", "MODEL_VERSION=integration-test",
        "-e", "UVICORN_WORKERS=1",
        IMAGE,
    ], cwd=repository_root, check=True, capture_output=True)

    base_url = f"http://127.0.0.1:{CONTAINER_PORT}"
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base_url}/ready", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            logs = subprocess.run(["docker", "logs", container_name],
                                  capture_output=True, text=True)
            pytest.fail(f"container never became ready. Logs:\n{logs.stdout}\n{logs.stderr}")
        yield base_url
    finally:
        subprocess.run(["docker", "stop", container_name], capture_output=True)


def test_container_reports_ready_and_alive(running_container):
    """Both probes answer, and they answer different questions."""
    with urllib.request.urlopen(f"{running_container}/health") as response:
        assert json.load(response)["status"] == "alive"
    with urllib.request.urlopen(f"{running_container}/ready") as response:
        body = json.load(response)
    assert body["status"] == "ready"
    assert body["model_version"] == "integration-test"


def test_container_predicts_and_reports_its_version(running_container):
    """The response carries a probability and says which model produced it."""
    status, body = http_post(f"{running_container}/predict", VALID_ROW)
    assert status == 200
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "integration-test"


def test_container_rejects_a_row_outside_the_schema(running_container):
    """A temperature no machine reaches must be refused, not scored.

    This is the contract the service promises callers, tested through the image rather
    than through the app object, because a validation layer that works in-process and not
    in the container is a distinction production will find for you.
    """
    impossible = {**VALID_ROW, "temp_c": 999.0}
    status, body = http_post(f"{running_container}/predict", impossible)
    assert status == 422
    assert "detail" in body


def test_container_batches_and_caps_the_batch(running_container):
    """100 rows are accepted in one call; 101 are refused by the schema."""
    status, body = http_post(f"{running_container}/predict/batch",
                             {"rows": [VALID_ROW] * 100})
    assert status == 200
    assert len(body["probabilities"]) == 100

    too_many, _ = http_post(f"{running_container}/predict/batch",
                            {"rows": [VALID_ROW] * 101})
    assert too_many == 422
