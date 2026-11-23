"""Check anonymous GOES S3 access and inspect one full-disk ABI scan per band."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import boto3
import numpy as np
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TMP = PROJECT_ROOT / "data" / "raw_tmp"
PREFIX = "ABI-L1b-RadF/2024/131/03/"
BUCKETS = ("noaa-goes16", "noaa-goes18")
BANDS = (13, 2)
CORNER_SIZE = 50
VNIR_BOX_SIZE = 200


def list_keys(s3: Any, bucket: str) -> list[str]:
    """List objects under the requested hour prefix and print a few examples."""
    print(f"\nS3 listing: s3://{bucket}/{PREFIX}")
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    status_code = "unknown"
    for page in paginator.paginate(Bucket=bucket, Prefix=PREFIX):
        status_code = str(page.get("ResponseMetadata", {}).get("HTTPStatusCode", "unknown"))
        keys.extend(item["Key"] for item in page.get("Contents", []))

    print(f"  Response: HTTP {status_code}; {len(keys)} object(s)")
    for key in keys[:3]:
        print(f"  Sample: s3://{bucket}/{key}")
    if not keys:
        print("  No objects found for this prefix.")
    return keys


def value_distribution(values: np.ndarray) -> str:
    """Format a compact value/count distribution, including any NaNs."""
    unique, counts = np.unique(values, return_counts=True)
    return ", ".join(
        f"{value!r}: {int(count)}" for value, count in zip(unique, counts)
    ) or "(empty)"


def variable_attribute(variable: xr.DataArray, name: str) -> Any:
    """Return a NetCDF attribute from decoded attrs or xarray's encoding."""
    return variable.attrs.get(name, variable.encoding.get(name, "<not present>"))


def inspect_corner(ds: xr.Dataset, label: str, *, raw: bool) -> None:
    rad = ds["Rad"]
    dqf = ds["DQF"]
    if rad.ndim != 2 or dqf.ndim != 2:
        raise ValueError(f"Expected 2-D Rad and DQF arrays, got {rad.dims} and {dqf.dims}")

    row_dim, col_dim = rad.dims
    corner_indexers = {
        row_dim: slice(0, CORNER_SIZE),
        col_dim: slice(0, CORNER_SIZE),
    }
    radiance = np.asarray(rad.isel(corner_indexers).values)
    flags = np.asarray(dqf.isel(corner_indexers).values)
    nan_count = int(np.isnan(radiance).sum()) if np.issubdtype(radiance.dtype, np.number) else 0
    finite_count = (
        int(np.isfinite(radiance).sum())
        if np.issubdtype(radiance.dtype, np.number)
        else radiance.size - nan_count
    )
    if nan_count == radiance.size:
        behavior = "all NaN (decoded as masked/fill pixels)"
    elif nan_count:
        behavior = f"mixed: {nan_count} NaN and {finite_count} finite numeric radiance value(s)"
    elif raw:
        behavior = f"unscaled numeric codes (no NaNs; {finite_count} value(s))"
    else:
        behavior = f"numeric radiances (no NaNs; {finite_count} finite value(s))"

    mode = "mask_and_scale=False" if raw else "mask_and_scale=True (default)"
    print(f"  {label} corner, {mode}, Rad[0:{CORNER_SIZE}, 0:{CORNER_SIZE}]:")
    print(f"    Radiance behavior: {behavior}")
    if raw:
        print(f"    Stored Rad value distribution: {value_distribution(radiance)}")
    print(f"    DQF distribution: {value_distribution(flags)}")


def describe_dataset(path: Path, band: int) -> None:
    print(f"\n{'=' * 78}\nBand C{band:02d}: {path.name}")
    for mask_and_scale in (True, False):
        with xr.open_dataset(
            path,
            engine="h5netcdf",
            mask_and_scale=mask_and_scale,
        ) as ds:
            print(
                f"\n  mask_and_scale={mask_and_scale}; dimensions: "
                f"{dict(ds.sizes)}"
            )
            print("  Diagnostic variables: Rad, DQF")
            for name in ("Rad", "DQF"):
                if name not in ds:
                    raise KeyError(f"{path.name} does not contain expected variable {name!r}")
                variable = ds[name]
                print(
                    f"  {name}: dims={variable.dims}, dtype={variable.dtype}, "
                    f"_FillValue={variable_attribute(variable, '_FillValue')!r}, "
                    f"scale_factor={variable_attribute(variable, 'scale_factor')!r}"
                )
            inspect_corner(
                ds,
                "Off-disk",
                raw=not mask_and_scale,
            )

            if band == 2 and mask_and_scale:
                rad = ds["Rad"]
                row_dim, col_dim = rad.dims
                projection = ds["goes_imager_projection"].attrs
                semi_major_axis = float(projection["semi_major_axis"])
                satellite_height = float(projection["perspective_point_height"])
                satellite_lon = float(projection["longitude_of_projection_origin"])
                target_lon = -85.0
                longitude_delta = np.deg2rad(target_lon - satellite_lon)
                target_x = np.arctan2(
                    semi_major_axis * np.sin(longitude_delta),
                    satellite_height
                    + semi_major_axis
                    - semi_major_axis * np.cos(longitude_delta),
                )
                center_row = int(
                    np.argmin(np.abs(np.asarray(rad.coords[row_dim].values)))
                )
                center_col = int(
                    np.argmin(
                        np.abs(np.asarray(rad.coords[col_dim].values) - target_x)
                    )
                )
                half_box = VNIR_BOX_SIZE // 2
                box_indexers = {
                    row_dim: slice(center_row - half_box, center_row + half_box),
                    col_dim: slice(center_col - half_box, center_col + half_box),
                }
                radiance = np.asarray(rad.isel(box_indexers).values)
                flags = np.asarray(ds["DQF"].isel(box_indexers).values)
                finite = radiance[np.isfinite(radiance)]
                print(
                    f"\n  C02 central-disk nightside diagnostic box: "
                    f"{row_dim}[{center_row - half_box}:{center_row + half_box}], "
                    f"{col_dim}[{center_col - half_box}:{center_col + half_box}] "
                    f"({VNIR_BOX_SIZE} x {VNIR_BOX_SIZE} pixels; centered near "
                    f"0 degrees latitude, {target_lon:.1f} degrees longitude; "
                    "equatorial Pacific Ocean)"
                )
                if finite.size:
                    print(
                        f"    Decoded Rad statistics (finite pixels): "
                        f"min={float(np.min(finite)):.8g}, "
                        f"max={float(np.max(finite)):.8g}, "
                        f"median={float(np.median(finite)):.8g}, "
                        f"std={float(np.std(finite)):.8g}; "
                        f"finite={finite.size}/{radiance.size}"
                    )
                else:
                    print("    Decoded Rad statistics: no finite pixels in the box.")
                print(f"    Decoded DQF distribution: {value_distribution(flags)}")


def main() -> None:
    RAW_TMP.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="us-east-1",
    )
    downloaded: list[Path] = []

    try:
        bucket_keys = {bucket: list_keys(s3, bucket) for bucket in BUCKETS}
        goes16_keys = bucket_keys["noaa-goes16"]
        selected: dict[int, str] = {}
        for band in BANDS:
            marker = f"C{band:02d}_G16_"
            matches = [key for key in goes16_keys if marker in Path(key).name]
            if not matches:
                raise FileNotFoundError(
                    f"No Band {band:02d} GOES-16 object found under s3://"
                    f"noaa-goes16/{PREFIX}"
                )
            selected[band] = matches[0]

        print("\nDownloading one GOES-16 file per requested band:")
        for band, key in selected.items():
            path = RAW_TMP / Path(key).name
            downloaded.append(path)
            print(f"  s3://noaa-goes16/{key} -> {path.relative_to(PROJECT_ROOT)}")
            s3.download_file("noaa-goes16", key, str(path))

        for band, key in selected.items():
            describe_dataset(RAW_TMP / Path(key).name, band)
        print("\nDiagnostic summary complete.")
    finally:
        print("\nTemporary-file cleanup:")
        for path in downloaded:
            if path.exists():
                path.unlink()
                print(f"  Removed {path.relative_to(PROJECT_ROOT)}")
        if not downloaded:
            print("  No temporary files were downloaded.")


if __name__ == "__main__":
    main()
