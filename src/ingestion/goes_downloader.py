"""Scaffold for anonymous GOES ABI L1b access and ROI cropping."""

from pathlib import Path
from typing import Any


def list_goes_scans(
    bucket: str,
    prefix: str,
    *,
    s3_client: Any | None = None,
) -> list[str]:
    """List scan object keys from a public NOAA GOES S3 bucket."""
    raise NotImplementedError


def download_crop_and_cleanup(
    bucket: str,
    object_key: str,
    output_path: str | Path,
    roi_lat_lon: dict[str, float],
    *,
    s3_client: Any | None = None,
) -> Path:
    """Download a scan, crop it to the ROI, and remove its temporary full disk."""
    raise NotImplementedError
