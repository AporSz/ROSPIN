"""Scaffold for lag and autocorrelation-aware statistical tests."""

from typing import Any


def lag_cross_correlation(
    signal_a: Any,
    signal_b: Any,
    *,
    max_lag: int,
) -> tuple[Any, Any]:
    """Compute cross-correlation values and their corresponding lags."""
    raise NotImplementedError


def block_bootstrap(
    values: Any,
    statistic: Any,
    *,
    block_size: int,
    n_resamples: int,
    random_state: int | None = None,
) -> Any:
    """Estimate a statistic's sampling distribution using block resampling."""
    raise NotImplementedError


def circular_shift_permutation(
    signal: Any,
    reference: Any,
    statistic: Any,
    *,
    n_permutations: int,
    random_state: int | None = None,
) -> Any:
    """Test association against circular shifts that preserve autocorrelation."""
    raise NotImplementedError
