import re
import warnings
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from lcc_trend_analysis.observations.ceda import (
    STFCChilboltonCT75kDataFile,
)
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.utils import (
    DimensionalityError,
    DimensionalityWarning,
)
from lcc_trend_analysis.type_aliases import Timestamp


def _chilbolton_data_path() -> Path:
    return get_data_paths().chilbolton_ct75k


def get_data_file_for_date_chilbolton(date: Timestamp) -> Optional[Path]:
    """Get Chilbolton CT75k data file for a specific date.

    Prioritizes corrected files over raw data.

    Args:
        date (Timestamp): Date of observations

    Returns:
        Path | None: Path to data file if found, None otherwise
    """
    dir_path = _chilbolton_data_path() / date.strftime("%Y") / date.strftime("%m")

    # Prioritize corrected files
    path = (
        dir_path
        / f"cfarr-lidar-ct75k_chilbolton_{date.strftime('%Y%m%d')}_cor1.nc"
    )

    if path.exists():
        return path

    path = (
        dir_path / f"cfarr-lidar-ct75k_chilbolton_{date.strftime('%Y%m%d')}.nc"
    )

    if path.exists():
        return path
    
    
    path = dir_path / f"lidar-ct75k_chilbolton_{date.strftime('%Y%m%d')}.nc"

    if path.exists():
        return path

    return None


def iter_available_dates_chilbolton() -> Generator[Timestamp, None, None]:
    """Iterate over available dates for Chilbolton site.

    Yields:
        Timestamp: Available observation dates
    """
    # Pattern to match filenames ending in an 8-digit date + .nc
    date_regex = re.compile(r"(\d{8})\.nc$")
    collected_dates = set()
    for path in _chilbolton_data_path().rglob("*.nc"):
        match = date_regex.search(path.name)
        if match:
            date_str = match.group(1)
            try:
                date = pd.to_datetime(arg=date_str, format="%Y%m%d")
                if date not in collected_dates:
                    collected_dates.add(date)
                    yield date
            except ValueError:
                continue


def get_chilbolton_data_files() -> Generator[
    STFCChilboltonCT75kDataFile, None, None
]:
    """Generate STFCChilboltonCT75kDataFile objects for all available dates.

    Yields:
        STFCChilboltonCT75kDataFile: Data file wrapper for each available date
    """
    for date in iter_available_dates_chilbolton():
        data_file_path = get_data_file_for_date_chilbolton(date)
        if data_file_path is None:
            continue
        try:
            data_file = STFCChilboltonCT75kDataFile.from_source_data_file(
                    site_id="chilbolton",
                    site_name="Chilbolton",
                    date=date,
                    data_path=data_file_path,
                )
        except DimensionalityError as e:
            warnings.warn(str(e), DimensionalityWarning)
            continue
        yield data_file
