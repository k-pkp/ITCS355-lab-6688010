"""Lab 3 Task 4 — a traffic splitter with a rollback control.

    MODEL_A=http://127.0.0.1:8081 MODEL_B=http://127.0.0.1:8082 \\
        uvicorn loadtest.canary_proxy:app --port 8080

Two upstreams, a weight, and an admin route that changes the weight. That is all a
provider's traffic split is: SageMaker calls them production variants, Azure ML calls them
traffic percentages, Vertex AI calls it a traffic split. The vocabulary differs; the object
is this.

Slots are named "blue" and "green" on purpose. The detector downstream must decide that one
slot is worse from its metrics alone, and slot names that said "old" and "new" would hand
it the answer.

Every routed request is appended to a JSONL decision log with a timestamp, which is the
evidence that traffic actually moved when the rollback happened.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import httpx2 as httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

UPSTREAMS = {
    "blue": os.environ.get("MODEL_A", "http://127.0.0.1:8081"),
    "green": os.environ.get("MODEL_B", "http://127.0.0.1:8082"),
}
DECISION_LOG = Path(os.environ.get("CANARY_LOG", "reports/canary-decisions.jsonl"))

# Starting split, as the lab requires: 90% to the incumbent, 10% to the canary.
STATE: dict[str, object] = {"weights": {"blue": 90, "green": 10}}

app = FastAPI(title="ITCS355 canary proxy")


def choose_slot(weights: dict[str, int]) -> str:
    """Pick a slot at random, in proportion to its weight."""
    total_weight = sum(weights.values())
    draw = random.uniform(0, total_weight)
    running = 0.0
    for slot, weight in weights.items():
        running += weight
        if draw <= running:
            return slot
    return list(weights)[-1]


def append_decision(record: dict) -> None:
    """Append one routing decision to the log, creating the file on first write."""
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_LOG.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness of the proxy itself, independent of either upstream."""
    return {"status": "alive"}


@app.get("/admin/weights")
def read_weights() -> dict:
    """Report the split currently in force."""
    return {"weights": STATE["weights"]}


@app.post("/admin/weights")
async def set_weights(request: Request) -> dict:
    """Change the split. This is the rollback control.

    Returns the old and new weights with a timestamp so the report can show exactly when
    traffic was moved, rather than asserting that it was.
    """
    body = await request.json()
    previous = dict(STATE["weights"])
    STATE["weights"] = {slot: int(body.get(slot, 0)) for slot in UPSTREAMS}

    change = {
        "event": "weights_changed",
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "from": previous,
        "to": STATE["weights"],
    }
    append_decision(change)
    return change


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    """Route one prediction to a slot and record which slot served it."""
    payload = await request.json()
    slot = choose_slot(STATE["weights"])

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as client:
        upstream_response = await client.post(f"{UPSTREAMS[slot]}/predict", json=payload)
    latency_ms = (time.perf_counter() - started) * 1000

    body = upstream_response.json()
    append_decision({
        "event": "routed",
        "ts": time.time(),
        "slot": slot,
        "status": upstream_response.status_code,
        "latency_ms": round(latency_ms, 3),
        "probability": body.get("probability"),
        "model_version": body.get("model_version"),
    })

    # The slot travels back to the caller; the model version does not have to. A real
    # canary is invisible to the client, which is why detection has to come from metrics.
    return JSONResponse(status_code=upstream_response.status_code,
                        content=body, headers={"x-canary-slot": slot})
