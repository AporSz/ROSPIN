import os
import re
import warnings
import numpy as np
import pandas as pd
import requests
import xarray as xr

# Add your quiet baseline day and Event 3 dates once Lucian locks them
DATES = ["20240510", "20240511", "20240512", "20240514", "20240515"]
OUT_DIR = "data/swpc"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_XRS = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science"
BASE_SGPS = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/sgps-l2-avg1m"


def fetch_daily_nc(base_url: str, prefix: str, ymd: str) -> str:
    yyyy, mm = ymd[:4], ymd[4:6]
    folder_url = f"{base_url}/{yyyy}/{mm}/"
    html = requests.get(folder_url, timeout=30).text
    match = re.search(rf"({prefix}_g16_d{ymd}_v[\d-]+\.nc)", html)
    if not match:
        raise FileNotFoundError(f"No file found for {prefix} on {ymd}")
    fname = match.group(1)
    local_path = os.path.join(OUT_DIR, fname)
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


def main() -> None:
    records = []
    for idx, ymd in enumerate(DATES):
        xrs_file = fetch_daily_nc(BASE_XRS, "sci_xrsf-l2-avg1m", ymd)
        sgps_file = fetch_daily_nc(BASE_SGPS, "sci_sgps-l2-avg1m", ymd)

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
    out_path = os.path.join(OUT_DIR, "goes16_swpc_flux.parquet")
    full_df.to_parquet(out_path, index=False)

    print(f"\nSaved {len(full_df)} rows to {out_path}")
    print(full_df[["xrsb_flux", "proton_gt10", "proton_gt50", "proton_gt100"]].describe())
    print("Time range:", full_df["time"].min(), "->", full_df["time"].max())


if __name__ == "__main__":
    main()