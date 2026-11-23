"""Scaffold for EUMETSAT Data Store access to MSG/SEVIRI data."""

from datetime import datetime
from pathlib import Path


def test_connection() -> bool:
    """Check that EUMETSAT credentials and Data Store access are available."""
    raise NotImplementedError


def download_window(
    collection_id: str,
    start: datetime,
    end: datetime,
    output_dir: str | Path,
) -> list[Path]:
    """Download available EUMETSAT observations within a time window."""
    raise NotImplementedError
