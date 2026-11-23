import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import median_filter


def compute_scan_metrics(
    rad_2d: np.ndarray,
    dqf_2d: np.ndarray,
    sigma_thresh: float = 5.0,
    filter_size: int = 3,
) -> dict:
    """Compute v0 snow and DQF metrics on a single native-grid (y, x) scan."""
    valid_on_disk = np.isfinite(dqf_2d) & (dqf_2d >= 0)
    n_valid = int(valid_on_disk.sum())

    if n_valid == 0:
        return {
            "n_valid_on_disk": 0,
            "n_good_dqf0": 0,
            "dqf_nonzero_frac": np.nan,
            "dqf_1_frac": np.nan,
            "dqf_2_frac": np.nan,
            "dqf_3_frac": np.nan,
            "dqf_4_frac": np.nan,
            "bg_noise_mad": np.nan,
            "outlier_count_5sig": 0,
            "outlier_rate_ppm": np.nan,
        }

    dqf_valid = dqf_2d[valid_on_disk]
    dqf_nonzero_frac = float(np.mean(dqf_valid != 0))
    dqf_counts = {code: float(np.mean(dqf_valid == code)) for code in (1, 2, 3, 4)}

    # Keep only DQF == 0 finite pixels before spatial filtering
    good_mask = valid_on_disk & (dqf_2d == 0) & np.isfinite(rad_2d)
    n_good = int(good_mask.sum())
    if n_good < 25:
        return {
            "n_valid_on_disk": n_valid,
            "n_good_dqf0": n_good,
            "dqf_nonzero_frac": dqf_nonzero_frac,
            "dqf_1_frac": dqf_counts[1],
            "dqf_2_frac": dqf_counts[2],
            "dqf_3_frac": dqf_counts[3],
            "dqf_4_frac": dqf_counts[4],
            "bg_noise_mad": np.nan,
            "outlier_count_5sig": 0,
            "outlier_rate_ppm": np.nan,
        }

    # Fill masked pixels with scene median so edges don't bleed into the 3x3 median filter
    fill_val = float(np.nanmedian(rad_2d[good_mask]))
    rad_filled = np.where(good_mask, rad_2d, fill_val)

    local_med = median_filter(rad_filled, size=filter_size, mode="nearest")
    residual = rad_filled - local_med
    res_good = residual[good_mask]

    med_res = float(np.median(res_good))
    mad_sigma = float(1.4826 * np.median(np.abs(res_good - med_res)))

    if mad_sigma == 0.0:
        outlier_count = 0
    else:
        # Positive spikes (SEU bright snow); use np.abs(res_good - med_res) for two-sided
        outlier_count = int(np.sum((res_good - med_res) > (sigma_thresh * mad_sigma)))

    outlier_rate_ppm = float((outlier_count / n_good) * 1e6)

    return {
        "n_valid_on_disk": n_valid,
        "n_good_dqf0": n_good,
        "dqf_nonzero_frac": dqf_nonzero_frac,
        "dqf_1_frac": dqf_counts[1],
        "dqf_2_frac": dqf_counts[2],
        "dqf_3_frac": dqf_counts[3],
        "dqf_4_frac": dqf_counts[4],
        "bg_noise_mad": mad_sigma,
        "outlier_count_5sig": outlier_count,
        "outlier_rate_ppm": outlier_rate_ppm,
    }


def process_abi_dataset(
    ds: xr.Dataset,
    satellite: str,
    band: str,
    c02_nightside_max_std: float = 2.5,
) -> pd.DataFrame:
    """Run compute_scan_metrics across all time steps of a cropped ABI dataset."""
    times = pd.to_datetime(ds["time"].values, utc=True)
    rows = []

    for idx, ts in enumerate(times):
        rad = ds["Rad"].isel(time=idx).values
        dqf = ds["DQF"].isel(time=idx).values
        metrics = compute_scan_metrics(rad, dqf)

        # Guard nightside dark-target C02 against daytime solar reflection/clouds
        is_nightside = True
        if band.upper() == "C02":
            valid_rad = rad[np.isfinite(dqf) & (dqf == 0) & np.isfinite(rad)]
            is_nightside = bool(valid_rad.size > 0 and np.std(valid_rad) < c02_nightside_max_std)

        rows.append(
            {
                "time": ts,
                "satellite": satellite,
                "band": band.upper(),
                "is_nightside": is_nightside,
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)