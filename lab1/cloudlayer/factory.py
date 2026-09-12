"""Adapter selection. The only place that maps CLOUD_PROVIDER to an implementation."""
from __future__ import annotations

from cloudlayer.base import CloudAdapter, LocalAdapter


def get_adapter(config) -> CloudAdapter:
    """Return the adapter for the configured provider.

        The only place in the codebase that maps CLOUD_PROVIDER to an implementation, so
        that swapping providers is one environment variable rather than an edit.
        """
    provider = (config.provider or "local").lower()
    if provider == "local":
        return LocalAdapter(config)
    if provider == "aws":
        from cloudlayer.aws import AwsAdapter
        return AwsAdapter(config)
    if provider == "azure":
        from cloudlayer.azure import AzureAdapter
        return AzureAdapter(config)
    if provider == "gcp":
        from cloudlayer.gcp import GcpAdapter
        return GcpAdapter(config)
    raise ValueError(f"Unknown CLOUD_PROVIDER={provider!r}. Use aws, azure, gcp, or local.")
