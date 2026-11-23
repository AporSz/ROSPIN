"""Scaffold for detector snow metrics and synthetic-signal validation."""

from typing import Any


def dqf_fraction(dqf: Any, *, invalid_flags: tuple[int, ...] = (1, 2, 3)) -> float:
    """Calculate the fraction of samples carrying a selected DQF value."""
    raise NotImplementedError


def local_median_outliers(
    radiance: Any,
    *,
    sigma_threshold: float = 5.0,
    window_size: int = 5,
) -> Any:
    """Identify local-median residuals exceeding the requested sigma threshold."""
    raise NotImplementedError


def dark_target_noise(radiance: Any, mask: Any | None = None) -> float:
    """Summarize noise over a dark-target region or supplied pixel mask."""
    raise NotImplementedError


def inject_synthetic_snow(
    radiance: Any,
    *,
    coordinates: Any,
    amplitude: float,
) -> Any:
    """Return a copy of radiance with synthetic spike anomalies injected."""
    raise NotImplementedError
