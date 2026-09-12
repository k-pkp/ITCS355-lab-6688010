"""Where the providers disagree about HTTP paths.

The service exposes /health, /ready, /predict and /predict/batch, and it exposes them the
same way everywhere. Each managed endpoint service then insists on its own names:

    SageMaker   GET /ping          POST /invocations     (fixed; the contract of the image)
    Azure ML    liveness route named on the deployment, scoring route on the endpoint
    Vertex AI   health and predict routes declared on the Model resource

Renaming the service's routes to please one of them is what makes an image non-portable —
you would then need a different image per provider, which is the thing the three-layer
split exists to prevent. So the difference is recorded here, in Layer 3, and applied at
deploy time as configuration.
"""
from __future__ import annotations

ROUTES_BY_PROVIDER: dict[str, dict[str, str]] = {
    "aws": {
        # SageMaker's paths are not configurable, so the adapter puts a path rewrite in
        # front of the container rather than touching the image.
        "health_route": "/ping",
        "predict_route": "/invocations",
        "rewrite_required": "yes — /ping to /health, /invocations to /predict",
    },
    "azure": {
        "health_route": "/health",
        "predict_route": "/predict",
        "rewrite_required": "no — both routes are named on the deployment",
    },
    "gcp": {
        # Vertex AI takes these verbatim on the Model resource, so no rewrite is needed.
        "health_route": "/health",
        "predict_route": "/predict",
        "rewrite_required": "no — declared on the Model resource",
    },
    "local": {
        "health_route": "/health",
        "predict_route": "/predict",
        "rewrite_required": "no",
    },
}


def routes_for(provider: str) -> dict[str, str]:
    """Return the route configuration one provider's endpoint expects."""
    key = (provider or "local").lower()
    if key not in ROUTES_BY_PROVIDER:
        raise KeyError(f"No route configuration for provider {key!r}. Add it to cloudlayer/routes.py.")
    return ROUTES_BY_PROVIDER[key]
