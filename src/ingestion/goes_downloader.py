"""Download GOES ABI L1b scans, crop them to a geographic ROI, and cache them."""

from __future__ import annotations

import argparse
import math
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import xarray as xr
import yaml
from botocore import UNSIGNED
from botocore.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SCAN_START_RE = re.compile(r"_s(?P<stamp>\d{14})_e")
CADENCE = timedelta(minutes=10)
DOWNLOAD_ATTEMPTS = 3


@dataclass(frozen=True)
class Scan:
    band: int
    key: str
    start: datetime


@dataclass
class RunCounts:
    downloaded: int = 0
    skipped: int = 0


def parse_utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone (for example, Z)")
    return parsed.astimezone(timezone.utc)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must contain a YAML mapping")
    return config


def validate_run_config(
    config: dict[str, Any], event_name: str
) -> tuple[datetime, datetime, dict[str, float], list[int]]:
    try:
        goes = config["goes"]
        event = config["events"][event_name]
        roi = goes["roi_lat_lon"]
        bands = goes["bands"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing required GOES/event configuration: {exc}") from exc

    start = parse_utc_timestamp(event.get("start"), f"events.{event_name}.start")
    end = parse_utc_timestamp(event.get("end"), f"events.{event_name}.end")
    if end < start:
        raise ValueError(f"events.{event_name}.end must not be earlier than start")

    roi_keys = ("lat_min", "lat_max", "lon_min", "lon_max")
    if not isinstance(roi, dict):
        raise ValueError("goes.roi_lat_lon must be a mapping")
    missing = [key for key in roi_keys if roi.get(key) is None]
    if missing:
        raise ValueError(
            "goes.roi_lat_lon bounds must not be null; missing: "
            + ", ".join(missing)
        )
    try:
        bounds = {key: float(roi[key]) for key in roi_keys}
    except (TypeError, ValueError) as exc:
        raise ValueError("All goes.roi_lat_lon bounds must be numeric") from exc
    if not (-90 <= bounds["lat_min"] < bounds["lat_max"] <= 90):
        raise ValueError("Latitude bounds must be ordered within [-90, 90]")
    if not (-180 <= bounds["lon_min"] < bounds["lon_max"] <= 180):
        raise ValueError("Longitude bounds must be ordered within [-180, 180]")
    if not isinstance(bands, list) or not bands:
        raise ValueError("goes.bands must be a non-empty list")
    try:
        band_ids = [int(band) for band in bands]
    except (TypeError, ValueError) as exc:
        raise ValueError("goes.bands must contain integer band IDs") from exc
    if any(band_id < 1 or band_id > 16 for band_id in band_ids):
        raise ValueError("GOES ABI band IDs must be between 1 and 16")

    return start, end, bounds, band_ids


def hourly_prefixes(product: str, start: datetime, end: datetime) -> list[str]:
    prefix_hour = start.replace(minute=0, second=0, microsecond=0)
    last_hour = end.replace(minute=0, second=0, microsecond=0)
    prefixes: list[str] = []
    while prefix_hour <= last_hour:
        prefixes.append(
            f"{product}/{prefix_hour:%Y}/{prefix_hour:%j}/{prefix_hour:%H}/"
        )
        prefix_hour += timedelta(hours=1)
    return prefixes


def scan_start_time(key: str) -> datetime:
    match = SCAN_START_RE.search(key)
    if not match:
        raise ValueError(f"Could not parse ABI scan start timestamp from {key!r}")
    stamp = match.group("stamp")
    try:
        return datetime.strptime(stamp[:13], "%Y%j%H%M%S").replace(
            tzinfo=timezone.utc
        ) + timedelta(milliseconds=int(stamp[13]) * 100)
    except ValueError as exc:
        raise ValueError(f"Invalid ABI scan start timestamp in {key!r}") from exc


def list_goes_scans(
    bucket: str,
    prefix: str,
    *,
    s3_client: Any | None = None,
) -> list[str]:
    """List S3 object keys below one NOAA GOES prefix using anonymous access."""
    client = s3_client or boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="us-east-1",
    )
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(
            item["Key"]
            for item in page.get("Contents", [])
            if "Key" in item
        )
    return keys


def collect_event_scans(
    bucket: str,
    product: str,
    bands: list[int],
    start: datetime,
    end: datetime,
    *,
    s3_client: Any | None = None,
) -> dict[int, list[Scan]]:
    client = s3_client or boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="us-east-1",
    )
    found: dict[int, list[Scan]] = {band: [] for band in bands}
    prefixes = hourly_prefixes(product, start, end)
    print(f"Listing {len(prefixes)} hourly prefix(es) in s3://{bucket}/{product}/")
    for prefix in prefixes:
        keys = list_goes_scans(bucket, prefix, s3_client=client)
        for key in keys:
            if not key.endswith(".nc") or product not in key:
                continue
            match = re.search(r"C(?P<band>\d{2})_G\d{2}_", Path(key).name)
            if not match:
                continue
            band = int(match.group("band"))
            if band not in found:
                continue
            scan_time = scan_start_time(key)
            if start <= scan_time <= end:
                found[band].append(Scan(band=band, key=key, start=scan_time))

    for scans in found.values():
        scans.sort(key=lambda scan: (scan.start, scan.key))
    for band, scans in found.items():
        print(f"  C{band:02d}: {len(scans)} scan(s) in requested time window")
    return found


def _lat_lon_to_scan_angles(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    projection: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Project geodetic coordinates to ABI fixed-grid scan angles in radians."""
    height = float(projection["perspective_point_height"])
    semi_major = float(projection["semi_major_axis"])
    semi_minor = float(projection["semi_minor_axis"])
    lon_origin = np.deg2rad(float(projection["longitude_of_projection_origin"]))
    lat = np.deg2rad(latitudes)
    lon_delta = np.deg2rad(longitudes) - lon_origin

    eccentricity_squared = 1.0 - (semi_minor**2 / semi_major**2)
    geocentric_lat = np.arctan((semi_minor**2 / semi_major**2) * np.tan(lat))
    radius = semi_minor / np.sqrt(
        1.0 - eccentricity_squared * np.cos(geocentric_lat) ** 2
    )
    r1 = height - radius * np.cos(geocentric_lat) * np.cos(lon_delta)
    r2 = -radius * np.cos(geocentric_lat) * np.sin(lon_delta)
    r3 = radius * np.sin(geocentric_lat)
    x = np.arctan2(-r2, r1)
    y = np.arctan2(r3, np.sqrt(r1**2 + r2**2))
    return y, x


def roi_fixed_grid_bounds(
    roi_lat_lon: dict[str, float],
    projection: dict[str, Any],
    *,
    edge_samples: int = 101,
) -> tuple[float, float, float, float]:
    """Project a sampled geographic rectangle boundary to a fixed-grid box."""
    lat_min = roi_lat_lon["lat_min"]
    lat_max = roi_lat_lon["lat_max"]
    lon_min = roi_lat_lon["lon_min"]
    lon_max = roi_lat_lon["lon_max"]
    edge = np.linspace(0.0, 1.0, edge_samples)
    latitudes = np.concatenate(
        (
            np.full(edge_samples, lat_min),
            np.full(edge_samples, lat_max),
            lat_min + edge * (lat_max - lat_min),
            lat_min + edge * (lat_max - lat_min),
        )
    )
    longitudes = np.concatenate(
        (
            lon_min + edge * (lon_max - lon_min),
            lon_min + edge * (lon_max - lon_min),
            np.full(edge_samples, lon_min),
            np.full(edge_samples, lon_max),
        )
    )
    y, x = _lat_lon_to_scan_angles(latitudes, longitudes, projection)
    visible = np.isfinite(x) & np.isfinite(y)
    if not visible.any():
        raise ValueError("ROI does not project to finite GOES fixed-grid coordinates")
    return (
        float(np.min(y[visible])),
        float(np.max(y[visible])),
        float(np.min(x[visible])),
        float(np.max(x[visible])),
    )


def _coordinate_slice(
    coordinate: xr.DataArray, lower: float, upper: float
) -> slice:
    values = np.asarray(coordinate.values)
    included = np.flatnonzero((values >= lower) & (values <= upper))
    if included.size == 0:
        raise ValueError(
            f"ROI does not intersect coordinate {coordinate.name!r} "
            f"range [{float(np.min(values))}, {float(np.max(values))}]"
        )
    first, last = int(included.min()), int(included.max())
    return slice(first, last + 1)


def crop_dataset(
    dataset: xr.Dataset, roi_lat_lon: dict[str, float]
) -> xr.Dataset:
    required = ("Rad", "DQF", "t", "goes_imager_projection")
    missing = [name for name in required if name not in dataset]
    if missing:
        raise ValueError("GOES scan is missing required variables: " + ", ".join(missing))

    projection = dataset["goes_imager_projection"].attrs
    y_min, y_max, x_min, x_max = roi_fixed_grid_bounds(
        roi_lat_lon, projection
    )
    y_dim, x_dim = dataset["Rad"].dims
    y_slice = _coordinate_slice(dataset[y_dim], y_min, y_max)
    x_slice = _coordinate_slice(dataset[x_dim], x_min, x_max)
    cropped = dataset[["Rad", "DQF", "goes_imager_projection"]].isel(
        {y_dim: y_slice, x_dim: x_slice}
    )
    cropped["Rad"] = cropped["Rad"].astype(np.float32)
    cropped["DQF"] = cropped["DQF"].astype(np.float32)
    cropped = cropped.assign_coords(t=dataset["t"])
    cropped.attrs = dict(dataset.attrs)
    cropped.attrs["roi_lat_min"] = float(roi_lat_lon["lat_min"])
    cropped.attrs["roi_lat_max"] = float(roi_lat_lon["lat_max"])
    cropped.attrs["roi_lon_min"] = float(roi_lat_lon["lon_min"])
    cropped.attrs["roi_lon_max"] = float(roi_lat_lon["lon_max"])

    for name in cropped.variables:
        cropped[name].encoding = {}
    return cropped


def download_crop_and_cleanup(
    bucket: str,
    object_key: str,
    output_path: str | Path,
    roi_lat_lon: dict[str, float],
    *,
    s3_client: Any | None = None,
    raw_dir: str | Path | None = None,
) -> Path:
    """Download, crop, atomically save, then always remove the raw full-disk file."""
    client = s3_client or boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="us-east-1",
    )
    raw_directory = Path(raw_dir) if raw_dir else PROJECT_ROOT / "data" / "raw_tmp"
    destination = Path(output_path)
    raw_directory.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_temp = destination.with_name(destination.name + ".tmp")

    raw_handle = tempfile.NamedTemporaryFile(
        prefix="goes_", suffix=".nc", dir=raw_directory, delete=False
    )
    raw_path = Path(raw_handle.name)
    raw_handle.close()
    try:
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                client.download_file(bucket, object_key, str(raw_path))
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt == DOWNLOAD_ATTEMPTS:
                    break
                delay = min(2 ** (attempt - 1), 4)
                print(
                    f"  Download attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed "
                    f"({exc}); retrying in {delay}s"
                )
                time.sleep(delay)
        if last_error is not None:
            raise RuntimeError(
                f"Failed to download s3://{bucket}/{object_key} after "
                f"{DOWNLOAD_ATTEMPTS} attempts"
            ) from last_error

        with xr.open_dataset(raw_path, engine="h5netcdf") as source:
            cropped = crop_dataset(source, roi_lat_lon)
            encoding = {
                name: {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                    "_FillValue": np.float32(np.nan),
                }
                for name in ("Rad", "DQF")
            }
            cropped.to_netcdf(
                output_temp,
                engine="h5netcdf",
                encoding=encoding,
            )
        output_temp.replace(destination)
        return destination
    finally:
        raw_path.unlink(missing_ok=True)
        output_temp.unlink(missing_ok=True)


def _satellite_code(bucket: str) -> str:
    match = re.search(r"goes(\d{2})$", bucket)
    if not match:
        raise ValueError(
            f"Bucket {bucket!r} does not identify a GOES satellite (expected "
            "noaa-goes16 or noaa-goes18)"
        )
    return f"G{match.group(1)}"


def output_path_for_scan(
    processed_dir: Path,
    event_name: str,
    satellite: str,
    scan: Scan,
) -> Path:
    return (
        processed_dir
        / event_name
        / satellite
        / f"C{scan.band:02d}"
        / Path(scan.key).name
    )


def missing_cadences(
    scans: list[Scan], start: datetime, end: datetime
) -> list[datetime]:
    expected_count = max(
        1,
        math.ceil((end - start).total_seconds() / CADENCE.total_seconds()),
    )
    matched_slots: set[int] = set()
    for scan in scans:
        offset = (scan.start - start).total_seconds()
        slot = round(offset / CADENCE.total_seconds())
        expected_offset = slot * CADENCE.total_seconds()
        if (
            0 <= slot < expected_count
            and abs(offset - expected_offset) <= 120
        ):
            matched_slots.add(slot)
    return [
        start + index * CADENCE
        for index in range(expected_count)
        if index not in matched_slots
    ]


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _format_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def run_event(
    event_name: str,
    *,
    bucket: str,
    max_scans: int | None,
    config: dict[str, Any],
) -> RunCounts:
    start, end, roi, bands = validate_run_config(config, event_name)
    goes = config["goes"]
    product = goes.get("product")
    if not isinstance(product, str) or not product:
        raise ValueError("goes.product must be a non-empty string")
    if max_scans is not None and max_scans < 1:
        raise ValueError("--max-scans must be a positive integer")
    satellite = _satellite_code(bucket)

    paths = config.get("paths", {})
    raw_dir = PROJECT_ROOT / paths.get("raw_tmp", "data/raw_tmp")
    processed_dir = PROJECT_ROOT / paths.get("processed", "data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Event {event_name}: {start.isoformat()} through {end.isoformat()} "
        f"(UTC; inclusive)"
    )
    print(f"Bucket: {bucket}; bands: {', '.join(f'C{band:02d}' for band in bands)}")
    print(f"ROI: {roi}")
    client = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="us-east-1",
    )
    scans_by_band = collect_event_scans(
        bucket,
        product,
        bands,
        start,
        end,
        s3_client=client,
    )

    counts = RunCounts()
    full_window_scans: list[Scan] = []
    for band, scans in scans_by_band.items():
        full_window_scans.extend(scans)
        selected = scans[:max_scans] if max_scans is not None else scans
        band_counts = RunCounts()
        print(f"\nProcessing C{band:02d}: {len(selected)} selected scan(s)")
        for index, scan in enumerate(selected, start=1):
            destination = output_path_for_scan(
                processed_dir, event_name, satellite, scan
            )
            if destination.exists():
                band_counts.skipped += 1
                print(
                    f"  [{index}/{len(selected)}] cached: "
                    f"{destination.relative_to(PROJECT_ROOT)}"
                )
                continue
            print(
                f"  [{index}/{len(selected)}] {scan.start.isoformat()} "
                f"{Path(scan.key).name}"
            )
            download_crop_and_cleanup(
                bucket,
                scan.key,
                destination,
                roi,
                s3_client=client,
                raw_dir=raw_dir,
            )
            band_counts.downloaded += 1
        print(
            f"  C{band:02d} downloaded={band_counts.downloaded}, "
            f"skipped={band_counts.skipped}"
        )
        counts.downloaded += band_counts.downloaded
        counts.skipped += band_counts.skipped

    print("\n10-minute cadence audit (24-hour event slots; end boundary excluded):")
    for band, scans in scans_by_band.items():
        missing = missing_cadences(scans, start, end)
        if missing:
            shown_missing = missing[:20]
            remaining = len(missing) - len(shown_missing)
            print(
                f"  C{band:02d}: {len(missing)} missing cadence(s): "
                + ", ".join(
                    item.strftime("%Y-%m-%dT%H:%MZ") for item in shown_missing
                )
                + (f", ... ({remaining} more)" if remaining else "")
            )
        else:
            print(f"  C{band:02d}: no missing cadence(s)")

    processed_size = directory_size(processed_dir)
    print(
        f"\nRun totals: downloaded={counts.downloaded}, skipped={counts.skipped}"
    )
    print(f"Processed data size: {_format_size(processed_size)} ({processed_size} bytes)")
    leftovers = sorted(raw_dir.glob("*.nc"))
    if leftovers:
        print(
            "WARNING: raw NetCDF file(s) remain in the temporary directory: "
            + ", ".join(path.name for path in leftovers)
        )
    else:
        print("Temporary raw NetCDF files remaining: 0")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and crop GOES ABI L1b scans for a configured event."
    )
    parser.add_argument("--event", required=True, help="Event key from config/config.yaml")
    parser.add_argument(
        "--bucket",
        help="Public NOAA GOES bucket (defaults to the first configured bucket)",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        help="Maximum scans per band for smoke testing",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config()
        goes = config.get("goes", {})
        configured_buckets = goes.get("buckets")
        if not isinstance(configured_buckets, list) or not configured_buckets:
            raise ValueError("goes.buckets must be a non-empty list")
        bucket = args.bucket or configured_buckets[0]
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("--bucket must be a non-empty bucket name")
        run_event(
            args.event,
            bucket=bucket,
            max_scans=args.max_scans,
            config=config,
        )
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
