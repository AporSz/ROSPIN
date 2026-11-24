import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.analysis.metrics import compute_scan_metrics, process_abi_dataset


def test_clean_dark_target_baseline():
    """Verify low false-positive rate and accurate MAD noise floor on clean nightside C02."""
    rng = np.random.default_rng(42)
    shape = (200, 200)  # Matches Apor-Tamas's 200x200 nightside C02 diagnostic box
    clean_rad = rng.normal(loc=0.01, scale=0.55, size=shape).astype(np.float32)
    dqf = np.zeros(shape, dtype=np.int8)

    metrics = compute_scan_metrics(clean_rad, dqf)

    assert metrics["n_valid_on_disk"] == 40_000
    assert metrics["n_good_dqf0"] == 40_000
    assert metrics["dqf_nonzero_frac"] == 0.0
    assert np.isclose(metrics["bg_noise_mad"], 0.55 * 0.861, atol=0.02)
    # Normal >5-sigma positive tail on 40,000 Gaussian pixels is ~0-1 pixel
    assert metrics["outlier_count_5sig"] <= 3


def test_synthetic_snow_injection_recovery():
    """Verify >=96% spike recovery when isolated +8-sigma proton hits are injected."""
    rng = np.random.default_rng(123)
    shape = (200, 200)
    sigma_bg = 0.55
    rad = rng.normal(loc=0.01, scale=sigma_bg, size=shape).astype(np.float32)
    dqf = np.zeros(shape, dtype=np.int8)

    # Space injected pixels on a >=3px grid so 3x3 median filter isolates single-pixel SEUs
    n_injected = 50
    candidate_ys = np.arange(3, 197, 4)
    candidate_xs = np.arange(3, 197, 4)
    grid_y, grid_x = np.meshgrid(candidate_ys, candidate_xs, indexing="ij")
    coords = np.column_stack([grid_y.ravel(), grid_x.ravel()])
    chosen = coords[rng.choice(len(coords), size=n_injected, replace=False)]

    rad[chosen[:, 0], chosen[:, 1]] += 8.0 * sigma_bg

    metrics = compute_scan_metrics(rad, dqf)

    assert metrics["outlier_count_5sig"] >= 48
    expected_ppm = (metrics["outlier_count_5sig"] / 40_000) * 1e6
    assert np.isclose(metrics["outlier_rate_ppm"], expected_ppm)


def test_c13_ocean_thermal_gradient_with_snow():
    """Verify smooth spatial SST gradients in C13 IR do not inflate false positives."""
    rng = np.random.default_rng(7)
    ny, nx = 100, 100
    y_grad = np.linspace(80.0, 95.0, ny)[:, None]
    x_grad = np.linspace(-2.0, 2.0, nx)[None, :]
    noise = rng.normal(0.0, 0.12, size=(ny, nx))
    rad = (y_grad + x_grad + noise).astype(np.float32)
    dqf = np.zeros((ny, nx), dtype=np.int8)

    clean_metrics = compute_scan_metrics(rad, dqf)
    assert clean_metrics["outlier_count_5sig"] <= 2

    # Inject 10 isolated +8-sigma spikes on top of the thermal gradient
    for idx in range(10):
        rad[10 + idx * 8, 10 + idx * 8] += 8.0 * 0.12

    spiked_metrics = compute_scan_metrics(rad, dqf)
    assert spiked_metrics["outlier_count_5sig"] >= 9


def test_dqf_masking_and_offdisk_exclusion():
    """Verify off-disk pixels (-1 / NaN) are excluded and DQF 1-4 are tracked then masked."""
    rad = np.zeros((20, 20), dtype=np.float32)
    dqf = np.zeros((20, 20), dtype=np.float32)

    # 100 off-disk pixels (5 rows x 20 cols): half -1, half NaN
    dqf[0:3, :] = -1.0
    dqf[3:5, :] = np.nan
    rad[0:5, :] = np.nan

    # Flagged on-disk pixels across codes 1, 2, 3, 4 (total = 20 pixels out of 300 on-disk)
    dqf[5, 0:5] = 1.0
    dqf[5, 5:10] = 2.0
    dqf[5, 10:15] = 3.0
    dqf[5, 15:20] = 4.0
    rad[5, :] = 9999.0  # Extreme corrupted values that must be masked before median filter

    metrics = compute_scan_metrics(rad, dqf)

    assert metrics["n_valid_on_disk"] == 300
    assert metrics["n_good_dqf0"] == 280
    assert np.isclose(metrics["dqf_nonzero_frac"], 20 / 300)
    assert np.isclose(metrics["dqf_1_frac"], 5 / 300)
    assert np.isclose(metrics["dqf_2_frac"], 5 / 300)
    assert np.isclose(metrics["dqf_3_frac"], 5 / 300)
    assert np.isclose(metrics["dqf_4_frac"], 5 / 300)
    assert metrics["outlier_count_5sig"] == 0


def test_all_offdisk_scan_returns_nans_safely():
    """Verify scans with zero valid on-disk pixels return NaN metrics without raising."""
    rad = np.full((30, 30), np.nan, dtype=np.float32)
    dqf = np.full((30, 30), -1.0, dtype=np.float32)

    metrics = compute_scan_metrics(rad, dqf)

    assert metrics["n_valid_on_disk"] == 0
    assert metrics["n_good_dqf0"] == 0
    assert np.isnan(metrics["dqf_nonzero_frac"])
    assert np.isnan(metrics["bg_noise_mad"])
    assert np.isnan(metrics["outlier_rate_ppm"])


def test_process_abi_dataset_and_c02_nightside_guard():
    """Verify multi-scan xr.Dataset ingestion and C02 nightside vs. daytime flagging."""
    rng = np.random.default_rng(99)
    times = pd.date_range("2024-05-11T03:00:00Z", periods=2, freq="10min")

    # Scan 0: nightside dark target (std ~ 0.55); Scan 1: daytime bright clouds (std ~ 30.0)
    rad_night = rng.normal(0.0, 0.55, size=(50, 50)).astype(np.float32)
    rad_day = rng.normal(80.0, 30.0, size=(50, 50)).astype(np.float32)
    rad_stack = np.stack([rad_night, rad_day], axis=0)
    dqf_stack = np.zeros((2, 50, 50), dtype=np.int8)

    ds = xr.Dataset(
        data_vars={
            "Rad": (("time", "y", "x"), rad_stack),
            "DQF": (("time", "y", "x"), dqf_stack),
        },
        coords={"time": times.tz_localize(None)},
    )

    df = process_abi_dataset(ds, satellite="G16", band="C02")

    assert len(df) == 2
    assert list(df["satellite"]) == ["G16", "G16"]
    assert list(df["band"]) == ["C02", "C02"]
    assert bool(df.loc[0, "is_nightside"]) is True
    assert bool(df.loc[1, "is_nightside"]) is False
    assert "outlier_rate_ppm" in df.columns