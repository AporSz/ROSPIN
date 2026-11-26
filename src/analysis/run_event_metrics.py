from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.analysis.metrics import compute_scan_metrics


def _extract_scan_time(ds: xr.Dataset, path: Path) -> pd.Timestamp:
    if "time" in ds.coords or "time" in ds.variables:
        val = np.asarray(ds["time"].values).ravel()[0]
        return pd.to_datetime(val, utc=True)
    if "t" in ds.variables:
        val = np.asarray(ds["t"].values).ravel()[0]
        return pd.to_datetime(val, utc=True)
    if "time_coverage_start" in ds.attrs:
        return pd.to_datetime(ds.attrs["time_coverage_start"], utc=True)
    match = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", path.name)
    if match:
        yyyy, jjj, hh, mm, ss = match.groups()
        return pd.to_datetime(f"{yyyy}-{jjj} {hh}:{mm}:{ss}", format="%Y-%j %H:%M:%S", utc=True)
    raise ValueError(f"Could not determine timestamp for {path.name}")


def _extract_meta(ds: xr.Dataset, path: Path) -> tuple[str, str]:
    sat_match = re.search(r"_(G1[68])_", path.name)
    satellite = sat_match.group(1) if sat_match else str(ds.attrs.get("platform_ID", "G16"))

    band_match = re.search(r"C(02|13)", path.name)
    if band_match:
        band = f"C{band_match.group(1)}"
    elif "band_id" in ds.variables:
        band = f"C{int(np.asarray(ds['band_id'].values).ravel()[0]):02d}"
    else:
        band = "UNKNOWN"
    return satellite, band


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ABI snow metrics for processed event NetCDFs.")
    parser.add_argument("--event", default="event_1", help="Event name (e.g., event_1)")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory with cropped NetCDFs")
    parser.add_argument("--exports-dir", default="data/exports", help="Output directory for Parquet metrics")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    exports_dir = Path(args.exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(processed_dir.rglob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found under {processed_dir}")

    rows = []
    for nc_path in nc_files:
        with xr.open_dataset(nc_path, engine="netcdf4") as ds:
            satellite, band = _extract_meta(ds, nc_path)
            rad_var = ds["Rad"]
            dqf_var = ds["DQF"]

            if "time" in rad_var.dims:
                for idx in range(rad_var.sizes["time"]):
                    ts = pd.to_datetime(rad_var["time"].values[idx], utc=True)
                    rad_2d = rad_var.isel(time=idx).values
                    dqf_2d = dqf_var.isel(time=idx).values
                    m = compute_scan_metrics(rad_2d, dqf_2d)
                    m["time"] = ts
                    m["satellite"] = satellite
                    m["band"] = band
                    m["is_nightside"] = bool(np.nanmedian(rad_2d) < 2.0) if band == "C02" else True
                    rows.append(m)
            else:
                ts = _extract_scan_time(ds, nc_path)
                rad_2d = rad_var.values
                dqf_2d = dqf_var.values
                m = compute_scan_metrics(rad_2d, dqf_2d)
                m["time"] = ts
                m["satellite"] = satellite
                m["band"] = band
                m["is_nightside"] = bool(np.nanmedian(rad_2d) < 2.0) if band == "C02" else True
                rows.append(m)

    df = pd.DataFrame(rows).sort_values(["band", "time"]).reset_index(drop=True)
    out_path = exports_dir / f"{args.event}_G16_metrics.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} scan rows to {out_path}")
    print(df.groupby("band")[["outlier_rate_ppm", "dqf_nonzero_frac", "bg_noise_mad"]].describe())


if __name__ == "__main__":
    main()