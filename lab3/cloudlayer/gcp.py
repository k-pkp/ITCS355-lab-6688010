"""GCP adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install google-cloud-storage google-cloud-aiplatform
Docs: storage.Client for GCS; Artifact Registry push goes through `docker push` after
      `gcloud auth configure-docker <region>-docker.pkg.dev`.

Hints for Lab 1:
  * BLOB_URI looks like gs://bucket/prefix — parse it here, never in src/.
  * Artifact Registry paths are region-scoped:
        <region>-docker.pkg.dev/<project>/<repo>/<image>
    A common first failure is pushing to gcr.io out of habit; it is a different service.
  * push_image must return the digest reference, not the tag.
  * GCP calls them labels, not tags, and they must be lowercase with no spaces.
    cfg.tags(1) already satisfies that constraint — do not "improve" the values.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from cloudlayer.base import CloudAdapter


def split_gcs_uri(blob_uri: str) -> tuple[str, str]:
    """Split a gs://bucket/prefix URI into its bucket name and its prefix.

    The prefix is returned without a leading or trailing slash so that callers can
    join it to a key with a single "/". A URI with no prefix returns an empty prefix.
    """
    if not blob_uri.startswith("gs://"):
        raise ValueError(f"BLOB_URI must start with gs:// — got {blob_uri!r}")

    without_scheme = blob_uri[len("gs://"):]
    bucket_name, _, prefix = without_scheme.partition("/")
    if not bucket_name:
        raise ValueError(f"BLOB_URI has no bucket name — got {blob_uri!r}")
    return bucket_name, prefix.strip("/")


class GcpAdapter(CloudAdapter):
    """Google Cloud implementation of the eleven-method portability seam."""

    def _storage_client(self):
        """Return a GCS client bound to the configured project.

        Imported inside the method so that a machine without the GCP SDK installed can
        still import this module — the factory only reaches here when CLOUD_PROVIDER=gcp.
        """
        from google.cloud import storage

        return storage.Client(project=self.config.project_id)

    def _bucket_and_prefix(self) -> tuple[str, str]:
        """Resolve BLOB_URI into the bucket and prefix this adapter writes under."""
        return split_gcs_uri(self.config.blob_uri)

    def _object_name(self, key: str) -> str:
        """Turn a caller's key into the full object name inside the bucket."""
        _, prefix = self._bucket_and_prefix()
        clean_key = key.strip("/")
        if prefix:
            return f"{prefix}/{clean_key}"
        return clean_key

    def upload(self, local_path: str, key: str) -> str:
        """Upload one local file under BLOB_URI and return its gs:// URI."""
        bucket_name, _ = self._bucket_and_prefix()
        object_name = self._object_name(key)

        client = self._storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(local_path)

        return f"gs://{bucket_name}/{object_name}"

    def download(self, uri: str, local_path: str) -> None:
        """Fetch a gs:// object to a local path, creating parent directories."""
        bucket_name, object_name = split_gcs_uri(uri)
        if not object_name:
            raise ValueError(f"URI names a bucket but no object — got {uri!r}")

        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        client = self._storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.download_to_filename(str(destination))

    def _registry_host(self) -> str:
        """Return the Artifact Registry hostname implied by CONTAINER_REGISTRY.

        CONTAINER_REGISTRY looks like <region>-docker.pkg.dev/<project>/<repo>; Docker
        authenticates per host, so the first path segment is what we log in against.
        """
        return self.config.container_registry.split("/")[0]

    def _access_token(self) -> str:
        """Mint a short-lived OAuth access token from the ambient GCP credentials.

        This is the same credential `gcloud auth configure-docker` would install, but it
        does not require the gcloud CLI to be on PATH, and the token expires in an hour
        rather than sitting in the Docker config forever.
        """
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token

    def _docker_login(self) -> None:
        """Authenticate the local Docker client against Artifact Registry."""
        subprocess.run(
            [
                "docker", "login",
                "-u", "oauth2accesstoken",
                "--password-stdin",
                f"https://{self._registry_host()}",
            ],
            input=self._access_token(),
            text=True,
            check=True,
            capture_output=True,
        )

    def push_image(self, local_tag: str) -> str:
        """Push a locally built image to Artifact Registry and return its digest reference.

        The returned reference is digest-pinned (repo@sha256:...) because a tag can be
        moved onto a different image later, and a moved tag is exactly the failure the
        lab is trying to make impossible.
        """
        image_name, _, tag = local_tag.partition(":")
        if not tag:
            tag = "latest"
        remote_tag = f"{self.config.container_registry}/{image_name}:{tag}"

        self._docker_login()
        subprocess.run(["docker", "tag", local_tag, remote_tag], check=True)
        subprocess.run(["docker", "push", remote_tag], check=True)

        inspect_result = subprocess.run(
            ["docker", "inspect", "--format={{json .RepoDigests}}", remote_tag],
            capture_output=True,
            text=True,
            check=True,
        )
        repo_digests = json.loads(inspect_result.stdout.strip())

        repository = f"{self.config.container_registry}/{image_name}"
        for digest_reference in repo_digests:
            if digest_reference.startswith(repository + "@"):
                return digest_reference

        raise RuntimeError(
            f"Pushed {remote_tag} but Docker reported no digest for {repository}. "
            f"RepoDigests was {repo_digests!r}."
        )

    # --- Lab 2: Vertex AI custom training and Model Registry ---------------------

    def _vertex(self):
        """Initialise the Vertex AI SDK for this project and region.

        Imported here rather than at module scope so that Lab 1, which needs only storage,
        does not require the much heavier aiplatform package to be installed.

        A missing SDK is turned into an explicit failure rather than allowed to surface as
        a bare ImportError, because one caller of this is `teardown`. "Teardown reported
        success, resources remain" is the expensive version of this mistake, and the
        difference between the two outcomes has to be unmistakable in the output.
        """
        try:
            from google.cloud import aiplatform
        except ImportError as error:
            raise RuntimeError(
                "google-cloud-aiplatform is not installed, so Vertex AI resources could "
                "NOT be inspected. This is not a clean result — nothing was checked and "
                "nothing was deleted. Install it with `pip install google-cloud-aiplatform` "
                "and run this again."
            ) from error

        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region,
            staging_bucket=self.config.blob_uri,
        )
        return aiplatform

    def submit_training(self, image_uri: str, arguments: dict[str, Any]) -> str:
        """Start a Vertex custom training job from an already-pushed image.

        Four things go in — an image, a command, a machine type, and an identity — which is
        the same four every provider wants under different parameter names. `args` carries
        the training script's own flags plus, optionally, `machine_type`, `service_account`
        and `use_spot`.

        Expect the first submission to fail on permissions. The identity that *submits* the
        job is your CLI identity; the identity that *runs* it is the service account named
        below, and it needs its own read access to BLOB_URI. They are different principals,
        and every provider gets students here.
        """
        aiplatform = self._vertex()

        machine_type = arguments.pop("machine_type", "e2-standard-4")
        service_account = arguments.pop("service_account", None)
        use_spot = arguments.pop("use_spot", True)
        display_name = arguments.pop("display_name", "itcs355-training")

        command_arguments: list[str] = []
        for flag, value in arguments.items():
            command_arguments.extend([f"--{flag.replace('_', '-')}", str(value)])

        job = aiplatform.CustomContainerTrainingJob(
            display_name=display_name,
            container_uri=image_uri,
            labels=self.config.tags(2),
        )
        job.run(
            arguments=command_arguments,
            machine_type=machine_type,
            replica_count=1,
            service_account=service_account,
            # Spot on Vertex is "restart_job_on_worker_restart" plus a spot strategy; the
            # training script checkpoints, so an interruption costs minutes, not the run.
            restart_job_on_worker_restart=use_spot,
            sync=False,
        )
        return job.resource_name

    def wait_training(self, job_id: str) -> dict[str, Any]:
        """Block until a submitted training job reaches a terminal state."""
        aiplatform = self._vertex()

        job = aiplatform.CustomTrainingJob.get(resource_name=job_id)
        job.wait()
        return {
            "job_id": job_id,
            "state": str(job.state),
            "error": getattr(job, "error", None),
        }

    def register_model(self, model_uri: str, name: str) -> str:
        """Upload a model artifact to the Vertex Model Registry and return its version.

        The health and predict routes come from cloudlayer.routes, not from the serving
        image: Vertex wants them declared on the Model resource, and absorbing that here is
        what keeps one image deployable on all three providers.
        """
        aiplatform = self._vertex()
        from cloudlayer.routes import routes_for

        routes = routes_for("gcp")
        model = aiplatform.Model.upload(
            display_name=name,
            artifact_uri=model_uri,
            serving_container_image_uri=f"{self.config.container_registry}/itcs355-serve:latest",
            serving_container_predict_route=routes["predict_route"],
            serving_container_health_route=routes["health_route"],
            serving_container_ports=[8080],
            labels=self.config.tags(2),
        )
        model.wait()
        return model.version_id

    # --- Lab 3: Vertex AI endpoints ----------------------------------------------

    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        """Deploy a registered model version to a Vertex endpoint, creating it if needed."""
        aiplatform = self._vertex()

        existing = aiplatform.Endpoint.list(filter=f'display_name="{endpoint}"')
        if existing:
            target_endpoint = existing[0]
        else:
            target_endpoint = aiplatform.Endpoint.create(
                display_name=endpoint, labels=self.config.tags(3)
            )

        model = aiplatform.Model(model_name=model_ref)
        model.deploy(
            endpoint=target_endpoint,
            machine_type=instance,
            min_replica_count=1,
            max_replica_count=1,
            # Named so that a canary can be given a traffic weight by name later. Vertex
            # calls this a traffic split; SageMaker calls it production variants; Azure ML
            # calls it traffic percentages. Same object, three vocabularies.
            deployed_model_display_name=f"{endpoint}-{model_ref.split('/')[-1]}",
        )
        return target_endpoint.resource_name

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a deployed endpoint once and normalise the response.

        Vertex returns `predictions` as a list; SageMaker returns the container's raw body;
        Azure ML returns the scoring script's return value. Callers should not have to know
        that, so the shape is flattened to the service's own response here. This is the
        genuine leak the Lab 5 write-up is about — the abstraction cannot hide that each
        provider wraps the answer differently, only absorb it in one place.
        """
        aiplatform = self._vertex()

        target_endpoint = aiplatform.Endpoint(endpoint_name=endpoint)
        response = target_endpoint.predict(instances=[payload])
        first = response.predictions[0]
        if isinstance(first, dict):
            return first
        return {"probability": float(first)}

    # --- Lab 4: Cloud Monitoring --------------------------------------------------

    def emit_metric(self, name: str, value: float, unit: str = "None") -> None:
        """Write one custom metric point to Cloud Monitoring.

        The metric type is namespaced under custom.googleapis.com/itcs355 so the dashboard
        in monitoring/dashboard.json can query a stable prefix, and so teardown can find
        everything this course created.
        """
        import time

        from google.api import metric_pb2
        from google.cloud import monitoring_v3

        client = monitoring_v3.MetricServiceClient()
        project_path = f"projects/{self.config.project_id}"

        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/itcs355/{name}"
        series.resource.type = "global"
        series.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
        series.value_type = metric_pb2.MetricDescriptor.ValueType.DOUBLE
        for label_name, label_value in self.config.tags(4).items():
            series.metric.labels[label_name] = label_value

        now_seconds = int(time.time())
        point = monitoring_v3.Point({
            "interval": {"end_time": {"seconds": now_seconds}},
            "value": {"double_value": float(value)},
        })
        series.points = [point]

        client.create_time_series(name=project_path, time_series=[series])

    # --- Lab 5: managed LLM and teardown ------------------------------------------

    def generate(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a managed Gemini model once and return the text plus real token counts.

        Token counts come from the response's own usageMetadata. Counting words instead
        would produce a cost report built on a number nobody measured, and every provider
        tokenises differently enough for that to be wrong by tens of percent.
        """
        import time

        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=True, project=self.config.project_id, location=self.config.region
        )
        model_name = params.get("model", "gemini-2.5-flash")

        configuration = types.GenerateContentConfig(
            temperature=params.get("temperature", 0.0),
            max_output_tokens=params.get("max_output_tokens", 512),
            system_instruction=params.get("system_instruction"),
        )

        started = time.perf_counter()
        response = client.models.generate_content(
            model=model_name, contents=prompt, config=configuration
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = response.usage_metadata
        return {
            "response": response.text,
            "input_tokens": int(usage.prompt_token_count or 0),
            "output_tokens": int(usage.candidates_token_count or 0),
            "cached_tokens": int(getattr(usage, "cached_content_token_count", 0) or 0),
            "latency_ms": latency_ms,
            "model": model_name,
        }

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Delete every endpoint, model and training job carrying these labels.

        Object storage is deliberately NOT touched. The DVC remote and the training data
        live under BLOB_URI, they cost almost nothing, and a teardown that can delete the
        dataset is a teardown nobody will dare run — which is how endpoints end up billing
        over a weekend.

        Deletion on GCP is asynchronous. A successful return here means "accepted", not
        "gone"; scripts/teardown_verify.py re-checks, and the bill is the final word.
        """
        aiplatform = self._vertex()

        label_filter = " AND ".join(f'labels.{key}="{value}"' for key, value in tags.items())
        deleted: list[str] = []

        for endpoint in aiplatform.Endpoint.list(filter=label_filter):
            endpoint.undeploy_all()
            endpoint.delete(force=True)
            deleted.append(f"endpoint:{endpoint.resource_name}")

        for model in aiplatform.Model.list(filter=label_filter):
            model.delete()
            deleted.append(f"model:{model.resource_name}")

        for job in aiplatform.CustomTrainingJob.list(filter=label_filter):
            if not job.state or "SUCCEEDED" not in str(job.state):
                job.cancel()
            job.delete()
            deleted.append(f"training_job:{job.resource_name}")

        return deleted
