"""Scaffold for generating paper and presentation figures from exports."""

from pathlib import Path


def make_figures(
    exports_dir: str | Path = "data/exports",
    output_dir: str | Path = "figures",
) -> list[Path]:
    """Generate publication figures from clean exported analysis data."""
    raise NotImplementedError
