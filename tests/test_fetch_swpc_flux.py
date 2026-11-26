import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.ingestion.fetch_swpc_flux import (
    clean_flux,
    dates_from_config,
    extract_integral_protons,
)


def test_dates_from_config_expands_utc_windows_and_skips_nulls():
    """Verify configured UTC intervals expand to unique sorted YYYYMMDD strings while skipping nulls."""
    cfg = {
        "events": {
            "event_1": {
                "start": "2024-05-10T18:00:00Z",
                "end": "2024-05-11T18:00:00Z",
            },
            "event_2": {
                "start": "2024-05-14T12:00:00Z",
                "end": "2024-05-15T12:00:00Z",
            },
            "event_3": {
                "start": "2024-06-08T00:00:00Z",
                "end": "2024-06-09T00:00:00Z",
            },
            "baseline": {
                "start": "2024-05-01T18:00:00Z",
                "end": "2024-05-02T18:00:00Z",
            },
            "unconfigured": {
                "start": None,
                "end": None,
            },
        }
    }

    assert dates_from_config(cfg) == [
        "20240501",
        "20240502",
        "20240510",
        "20240511",
        "20240514",
        "20240515",
        "20240608",
        "20240609",
    ]


def test_dates_from_config_rejects_inverted_window():
    """Verify end < start raises ValueError."""
    cfg = {
        "events": {
            "bad_event": {
                "start": "2024-05-12T00:00:00Z",
                "end": "2024-05-11T00:00:00Z",
            }
        }
    }
    with pytest.raises(ValueError, match="must not precede start"):
        dates_from_config(cfg)


def test_clean_flux_masks_negative_sentinels():
    """Verify negative fill values become NaN without mutating the source array."""
    raw = np.array([-9999.0, 0.0, 1.25e-5, -1.0e31], dtype=np.float32)
    cleaned = clean_flux(raw)

    assert raw[0] == -9999.0
    assert np.isnan(cleaned[0])
    assert cleaned[1] == 0.0
    assert np.isclose(cleaned[2], 1.25e-5)
    assert np.isnan(cleaned[3])


def test_extract_integral_protons_kev_and_telescope_averaging():
    """Verify differential keV integration + >100 MeV tail across two SGPS telescopes."""
    times = pd.date_range("2024-05-11T06:00:00Z", periods=3, freq="1min").tz_localize(None)

    # 3 channels in keV: [1000-5000] (<10 MeV), [10000-30000] (10-30 MeV), [50000-100000] (50-100 MeV)
    e_low = np.array([[1000.0, 10000.0, 50000.0], [1000.0, 10000.0, 50000.0]])
    e_high = np.array([[5000.0, 30000.0, 100000.0], [5000.0, 30000.0, 100000.0]])

    # Shape: (time=3, sensor=2, diff_chan=3), units in protons/(cm^2 s sr keV)
    diff_flux = np.zeros((3, 2, 3), dtype=np.float64)
    # t=0: sensor 0 and sensor 1 average to [0.1, 0.002, 0.0004] / keV
    diff_flux[0, 0, :] = [0.1, 0.001, 0.0002]
    diff_flux[0, 1, :] = [0.1, 0.003, 0.0006]
    # t=1: sensor 1 is negative fill (-9999), so sensor 0 [0.05, 0.001, 0.0001] is used
    diff_flux[1, 0, :] = [0.05, 0.001, 0.0001]
    diff_flux[1, 1, :] = [-9999.0, -9999.0, -9999.0]
    # t=2: all sentinel fill -> should produce NaN
    diff_flux[2, :, :] = -9999.0

    # Integral >100 MeV channel across 2 telescopes: shape (time=3, sensor=2)
    int_flux = np.array(
        [
            [4.0, 6.0],        # mean = 5.0 pfu
            [2.0, -9999.0],    # mean = 2.0 pfu
            [-9999.0, -9999.0],
        ],
        dtype=np.float64,
    )

    ds = xr.Dataset(
        data_vars={
            "AvgDiffProtonFlux": (
                ("time", "sensor", "diff_chan"),
                diff_flux,
                {"units": "protons/(cm^2 s sr keV)"},
            ),
            "AvgIntProtonFlux": (
                ("time", "sensor"),
                int_flux,
                {"units": "protons/(cm^2 s sr)"},
            ),
            "DiffProtonLowerEnergy": (
                ("sensor", "diff_chan"),
                e_low,
                {"units": "keV"},
            ),
            "DiffProtonUpperEnergy": (
                ("sensor", "diff_chan"),
                e_high,
                {"units": "keV"},
            ),
        },
        coords={"time": times},
    )

    df = extract_integral_protons(ds)

    # t=0:
    #   chan1 (10-30 MeV): 0.002 * 20_000 keV = 40.0
    #   chan2 (50-100 MeV): 0.0004 * 50_000 keV = 20.0
    #   gt100 = 5.0 -> gt50 = 25.0, gt10 = 65.0
    assert np.isclose(df.loc[0, "proton_gt100"], 5.0)
    assert np.isclose(df.loc[0, "proton_gt50"], 25.0)
    assert np.isclose(df.loc[0, "proton_gt10"], 65.0)

    # t=1:
    #   chan1: 0.001 * 20_000 = 20.0; chan2: 0.0001 * 50_000 = 5.0; gt100 = 2.0
    assert np.isclose(df.loc[1, "proton_gt100"], 2.0)
    assert np.isclose(df.loc[1, "proton_gt50"], 7.0)
    assert np.isclose(df.loc[1, "proton_gt10"], 27.0)

    # t=2: all missing -> NaN across all three integral outputs
    assert np.isnan(df.loc[2, "proton_gt10"])
    assert np.isnan(df.loc[2, "proton_gt50"])
    assert np.isnan(df.loc[2, "proton_gt100"])
