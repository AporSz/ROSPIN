from __future__ import annotations

import argparse
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_DATES = ["20240510", "20240511", "20240512", "20240514", "20240515"]
DATES = [
    "20240501", "20240502",  # baseline (2024-05-01T18:00Z -> 2024-05-02T18:00Z)
    "20240510", "20240511",  # event_1  (2024-05-10T18:00Z -> 2024-05-11T18:00Z)
    "20240514", "20240515",  # event_2  (2024-05-14T12:00Z -> 2024-05-15T12:00Z)
    "20240608", "20240609",  # event_3  (2024-06-08T00:00Z -> 2024-06-09T00:00Z)
]
OUT_DIR = "data/swpc"

BASE_XRS = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science"
BASE_SGPS = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/sgps-l2-avg1m"


def _parse_utc(value: str, label: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone offset (e.g., Z)")
    return parsed.astimezone(timezone.utc)


def dates_from_config(config_or_path: dict[str, Any] | str | Path = CONFIG_PATH) -> list[str]:
    """Expand configured non-null UTC event/baseline windows into sorted YYYYMMDD strings."""
    if isinstance(config_or_path, (str, Path)):
        with Path(config_or_path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    else:
        config = config_or_path

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    events = config.get("events", {})
    if not isinstance(events, dict):
        raise ValueError("config.events must be a mapping")

    collected: set[str] = set()
    for name, window in events.items():
        if not isinstance(window, dict):
            continue
        raw_start = window.get("start")
        raw_end = window.get("end")
        if raw_start is None or raw_end is None:
            continue
        if not isinstance(raw_start, str) or not isinstance(raw_end, str):
            raise ValueError(f"events.{name} start/end must be ISO-8601 strings or null")

        start_dt = _parse_utc(raw_start, f"events.{name}.start")
        end_dt = _parse_utc(raw_end, f"events.{name}.end")
        if end_dt < start_dt:
            raise ValueError(f"events.{name}.end must not precede start")

        day = start_dt.date()
        last_day = end_dt.date()
        while day <= last_day:
            collected.add(day.strftime("%Y%m%d"))
            day += timedelta(days=1)

    return sorted(collected)


def fetch_daily_nc(base_url: str, prefix: str, ymd: str, out_dir: str | Path = OUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    yyyy, mm = ymd[:4], ymd[4:6]
    folder_url = f"{base_url}/{yyyy}/{mm}/"
    resp = requests.get(folder_url, timeout=30)
    resp.raise_for_status()
    html = resp.text
    match = re.search(rf"({prefix}_g16_d{ymd}_v[\d-]+\.nc)", html)
    if not match:
        raise FileNotFoundError(f"No file found for {prefix} on {ymd}")
    fname = match.group(1)
    local_path = os.path.join(str(out_dir), fname)
    if not os.path.exists(local_path):
        r = requests.get(folder_url + fname, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path


def clean_flux(arr: np.ndarray) -> np.ndarray:
    """Replace negative NOAA sentinel fill values with NaN."""
    out = np.asarray(arr, dtype=np.float64).copy()
    out[out < 0] = np.nan
    return out


def _get_1d_channel_array(da: xr.DataArray) -> np.ndarray:
    """Reduce a channel-bound DataArray across any sensor/telescope dimension to 1D (channel,)."""
    for dim in da.dims:
        if "chan" not in dim.lower():
            da = da.isel({dim: 0})
    return np.asarray(da.values, dtype=np.float64)


def extract_integral_protons(ds_p: xr.Dataset, verbose: bool = False) -> pd.DataFrame:
    """Compute >10, >50, and >100 MeV integral proton fluxes from GOES-16 SGPS L2."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # 1. >100 MeV integral channel (average across +X and -X telescopes)
        int_flux = clean_flux(ds_p["AvgIntProtonFlux"].values)
        while int_flux.ndim > 1:
            int_flux = np.nanmean(int_flux, axis=-1)
        gt100 = int_flux

        # 2. Differential channels (average across telescope/sensor dimension)
        diff_da = ds_p["AvgDiffProtonFlux"]
        for dim in diff_da.dims:
            if dim != "time" and "chan" not in dim.lower():
                diff_da = diff_da.where(diff_da >= 0).mean(dim=dim)
        diff_flux = clean_flux(diff_da.values)  # shape: (time, channel)

    # 3. Separate lower and upper energy bin edges per channel
    e_low = _get_1d_channel_array(ds_p["DiffProtonLowerEnergy"])
    e_high = _get_1d_channel_array(ds_p["DiffProtonUpperEnergy"])

    flux_units = str(ds_p["AvgDiffProtonFlux"].attrs.get("units", "")).lower()
    bound_units = str(ds_p["DiffProtonLowerEnergy"].attrs.get("units", "")).lower()

    e_low_mev = e_low / 1000.0 if "kev" in bound_units else e_low
    delta_e = e_high - e_low
    if "kev" in flux_units and "mev" in bound_units:
        delta_e = delta_e * 1000.0
    elif "mev" in flux_units and "kev" in bound_units:
        delta_e = delta_e / 1000.0

    if verbose:
        print("SGPS differential lower edges (MeV):", np.round(e_low_mev, 2))
        print("SGPS flux units:", flux_units, "| bound units:", bound_units)

    # Integrate differential bins >= threshold and add the >100 MeV integral tail
    mask_gt50 = e_low_mev >= 45.0
    mask_gt10 = e_low_mev >= 9.0

    gt50 = np.nansum(diff_flux[:, mask_gt50] * delta_e[mask_gt50], axis=1) + gt100
    gt10 = np.nansum(diff_flux[:, mask_gt10] * delta_e[mask_gt10], axis=1) + gt100

    all_nan = np.all(np.isnan(diff_flux), axis=1) & np.isnan(gt100)
    gt50[all_nan] = np.nan
    gt10[all_nan] = np.nan

    return pd.DataFrame({
        "time": pd.to_datetime(ds_p["time"].values, utc=True),
        "proton_gt10": gt10,
        "proton_gt50": gt50,
        "proton_gt100": gt100,
    })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch GOES-16 XRS X-ray and SGPS integral proton fluxes for configured dates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        help="Explicit YYYYMMDD date strings (overrides config/config.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for NetCDF cache and Parquet export (overrides paths.swpc)",
    )
    args = parser.parse_args(argv)

    config: dict[str, Any] = {}
    if args.config.exists():
        with args.config.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
            if isinstance(loaded, dict):
                config = loaded

    if args.dates:
        target_dates = sorted({d.strip() for d in args.dates if d.strip()})
    else:
        configured_dates = dates_from_config(config) if config else []
        target_dates = configured_dates or list(DEFAULT_DATES)

    out_dir = args.out_dir or (PROJECT_ROOT / config.get("paths", {}).get("swpc", OUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching SWPC XRS & SGPS archives for {len(target_dates)} UTC day(s): {', '.join(target_dates)}")

    records = []
    for idx, ymd in enumerate(target_dates):
        xrs_file = fetch_daily_nc(BASE_XRS, "sci_xrsf-l2-avg1m", ymd, out_dir=out_dir)
        sgps_file = fetch_daily_nc(BASE_SGPS, "sci_sgps-l2-avg1m", ymd, out_dir=out_dir)

        with xr.open_dataset(xrs_file, engine="netcdf4") as ds_x, \
             xr.open_dataset(sgps_file, engine="netcdf4") as ds_p:

            xrsb_flag = ds_x["xrsb_flag"].values
            xrsa = clean_flux(ds_x["xrsa_flux"].values)
            xrsb = clean_flux(ds_x["xrsb_flux"].values)
            xrsb[xrsb_flag != 0] = np.nan

            df_x = pd.DataFrame({
                "time": pd.to_datetime(ds_x["time"].values, utc=True),
                "xrsa_flux": xrsa,
                "xrsb_flux": xrsb,
                "xrsb_flag": xrsb_flag,
            })

            df_p = extract_integral_protons(ds_p, verbose=(idx == 0))

            merged = pd.merge_asof(
                df_x.sort_values("time"),
                df_p.sort_values("time"),
                on="time",
                tolerance=pd.Timedelta("1min"),
                direction="nearest",
            )
            records.append(merged)

    full_df = pd.concat(records, ignore_index=True)
    out_path = out_dir / "goes16_swpc_flux.parquet"
    full_df.to_parquet(out_path, index=False)

    print(f"\nSaved {len(full_df)} rows to {out_path}")
    print(full_df[["xrsb_flux", "proton_gt10", "proton_gt50", "proton_gt100"]].describe())
    print("Time range:", full_df["time"].min(), "->", full_df["time"].max())


if __name__ == "__main__":
    main()
