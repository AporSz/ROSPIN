"""Generate paper and presentation figures from exported analysis tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def make_figures(
    exports_dir: str | Path = "data/exports",
    output_dir: str | Path = "figures",
    swpc_path: str | Path = "data/swpc/goes16_swpc_flux.parquet",
) -> list[Path]:
    """Generate publication figures from clean exported analysis data."""
    exports_dir = Path(exports_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_path = Path(__file__).with_name("plot_style.mplstyle")
    if style_path.exists():
        plt.style.use(style_path)

    flux_df = pd.read_parquet(swpc_path)
    flux_df["time"] = pd.to_datetime(flux_df["time"], utc=True)

    metrics_file = exports_dir / "event_1_G16_metrics.parquet"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Expected exported metrics at {metrics_file}")

    df = pd.read_parquet(metrics_file)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    df_c13 = df[df["band"] == "C13"].sort_values("time")
    df_c02 = df[(df["band"] == "C02") & (df["is_nightside"])].sort_values("time")

    t_start = df["time"].min() - pd.Timedelta("30min")
    t_end = df["time"].max() + pd.Timedelta("30min")
    flux_win = flux_df[(flux_df["time"] >= t_start) & (flux_df["time"] <= t_end)].sort_values("time")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    # Panel 1: GOES-16 XRS-B (0.1-0.8 nm) X-ray flux
    axes[0].plot(flux_win["time"], flux_win["xrsb_flux"], color="crimson", lw=1.2, label="XRS-B (0.1–0.8 nm)")
    axes[0].set_yscale("log")
    axes[0].axhline(1e-4, color="gray", ls="--", lw=0.8, label="X1.0 Flare")
    axes[0].set_ylabel("X-Ray Flux\n(W/m²)")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].set_title("Checkpoint 1 — Event 1 (GOES-16): Solar Flux vs. ABI L1b Snow & DQF Metrics")

    # Panel 2: GOES-16 SGPS Integral Proton Flux (>10, >50, >100 MeV)
    axes[1].plot(flux_win["time"], flux_win["proton_gt10"], color="darkorange", lw=1.2, label=">10 MeV")
    axes[1].plot(flux_win["time"], flux_win["proton_gt50"], color="royalblue", lw=1.2, label=">50 MeV")
    axes[1].plot(flux_win["time"], flux_win["proton_gt100"], color="purple", lw=1.2, label=">100 MeV")
    axes[1].set_yscale("log")
    axes[1].axhline(10.0, color="gray", ls="--", lw=0.8, label="NOAA S1 (10 pfu)")
    axes[1].set_ylabel("Proton Flux\n(pfu)")
    axes[1].legend(loc="upper right", frameon=True)

    # Panel 3: >5-sigma Spatial Outlier Rate (Type I SEU "Snow")
    if not df_c13.empty:
        axes[2].plot(df_c13["time"], df_c13["outlier_rate_ppm"], color="teal", marker=".", ms=3, lw=1.0, label="C13 IR (10.3 µm)")
    if not df_c02.empty:
        axes[2].plot(df_c02["time"], df_c02["outlier_rate_ppm"], color="navy", marker="o", ms=3.5, ls="none", label="C02 VNIR (Nightside)")
    axes[2].set_ylabel(">5σ Outliers\n(ppm)")
    axes[2].legend(loc="upper right", frameon=True)

    # Panel 4: DQF != 0 Fraction & Spatial MAD Noise Floor
    if not df_c13.empty:
        ax4_twin = axes[3].twinx()
        l1 = axes[3].plot(df_c13["time"], df_c13["dqf_nonzero_frac"] * 100.0, color="black", lw=1.1, label="C13 DQF != 0 (%)")
        l2 = ax4_twin.plot(df_c13["time"], df_c13["bg_noise_mad"], color="seagreen", ls="--", lw=1.0, label="C13 MAD Noise")
        axes[3].set_ylabel("DQF != 0 (%)")
        ax4_twin.set_ylabel("MAD Noise\n(radiance)", color="seagreen")
        lines = l1 + l2
        axes[3].legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True)

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[3].set_xlabel("UTC Time (2024-05-10 to 2024-05-11)")

    fig_path = output_dir / "checkpoint1_event1_timeseries.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")
    return [fig_path]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate figures from exported metrics.")
    parser.add_argument("--exports-dir", default="data/exports")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    make_figures(exports_dir=args.exports_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())