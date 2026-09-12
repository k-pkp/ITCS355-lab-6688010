"""AWS adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install boto3
Docs: S3 -> boto3 client("s3"); ECR -> boto3 client("ecr") for the auth token,
      then `docker push` through subprocess.

Hints for Lab 1:
  * BLOB_URI looks like s3://bucket/prefix — parse it here, never in src/.
  * ECR login expires. If a push that worked yesterday fails today, re-authenticate:
        aws ecr get-login-password --region $REGION | docker login --username AWS \
            --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
  * Return the DIGEST reference from push_image, not the tag. `docker inspect` or the
    push output gives you the sha256.
  * Tag the bucket objects and the ECR repository with cfg.tags(1).
"""
from __future__ import annotations

from typing import Any

from cloudlayer.base import CloudAdapter


def split_s3_uri(uri: str) -> tuple[str, str]:
    """Split an s3://bucket/prefix URI into its bucket and its key prefix."""
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {uri!r}")

    without_scheme = uri[len("s3://"):]
    bucket_name, _, key_prefix = without_scheme.partition("/")
    if not bucket_name:
        raise ValueError(f"URI has no bucket name: {uri!r}")
    return bucket_name, key_prefix.strip("/")


class AwsAdapter(CloudAdapter):
    """AWS implementation of the three methods Lab 5's swap test exercises.

    Only `upload`, `download` and `invoke` are implemented, which is exactly what Lab 5
    Task 4 Part B asks for: prove the seam is real, do not migrate the system. The rest
    stay unimplemented on purpose — writing them without an AWS account to run them
    against would produce code that has never executed, and the course is explicit that
    unrun code is not evidence.
    """

    def _client(self, service: str):
        """Return a boto3 client for one service, in the configured region."""
        import boto3

        return boto3.client(service, region_name=self.config.region)

    def upload(self, local_path: str, key: str) -> str:
        """Upload a file under BLOB_URI and return its s3:// URI."""
        bucket_name, key_prefix = split_s3_uri(self.config.blob_uri)
        object_key = f"{key_prefix}/{key.strip('/')}" if key_prefix else key.strip("/")

        self._client("s3").upload_file(local_path, bucket_name, object_key)
        return f"s3://{bucket_name}/{object_key}"

    def download(self, uri: str, local_path: str) -> None:
        """Fetch an S3 object to a local path, creating parent directories."""
        from pathlib import Path

        bucket_name, object_key = split_s3_uri(uri)
        if not object_key:
            raise ValueError(f"URI names a bucket but no object: {uri!r}")

        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._client("s3").download_file(bucket_name, object_key, str(destination))

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a SageMaker real-time endpoint once and normalise the response.

        SageMaker hands back the container's raw body, so what comes out is whatever
        service/app.py wrote — a JSON object with `probability` and `model_version`. Vertex
        instead wraps answers in a `predictions` list. Neither is wrong; they are simply
        different, and a caller should not have to know which cloud it is talking to. This
        is the leak the write-up is about: the seam can absorb the difference, but it
        cannot make it not exist.
        """
        import json

        response = self._client("sagemaker-runtime").invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=json.dumps(payload).encode(),
        )
        body = json.loads(response["Body"].read())
        if isinstance(body, dict):
            return body
        return {"probability": float(body)}

    def push_image(self, local_tag: str) -> str:
        """Push a local image to ECR. Not implemented: this project deploys on GCP."""
        raise NotImplementedError(
            "Not implemented: this project deploys on GCP. See GcpAdapter.push_image."
        )

    # submit_training / register_model  -> Lab 2 (SageMaker training job + model package group)
    # deploy / invoke                   -> Lab 3 (SageMaker real-time endpoint)
    # emit_metric                       -> Lab 4 (CloudWatch put_metric_data)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resourcegroupstaggingapi to find by tag)
