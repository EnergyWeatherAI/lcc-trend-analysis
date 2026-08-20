import warnings
from pathlib import Path
from typing import Generator

import pandas as pd

from lcc_trend_analysis.observations.arm import ARMDataFile
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.utils import (
    DimensionalityError,
    DimensionalityWarning,
)
from lcc_trend_analysis.type_aliases import (
    DataFrame,
    Timestamp,
)


def _arm_data_path() -> Path:
    return get_data_paths().arm


def collect_arm_data_files(
    arm_ids: str,
) -> Generator[tuple[Timestamp, Path], None, None]:
    """Iterate over available ARM ceilometer data files for a site.

    Args:
        arm_ids (str): String of comma-separated ARM site identifiers (e.g., "sgp.C1,sgp.B1")

    Yields:
        Generator[tuple[Timestamp, Path], None, None]: Generator of tuples of (date, path) for available data files
    """
    # Check if there are more ARM ids than just one associated with the site. A bit ugly, I admit.
    arm_ids_list = arm_ids.split(",")
    collected_dates = set()
    for arm_id in arm_ids_list:
        arm_site_id = arm_id.split(".")[0].lower()
        arm_subsite_id = arm_id.split(".")[1]

        for path in (
            _arm_data_path()
            / "ceil10m"
            / f"{arm_site_id}ceil10m{arm_subsite_id}.b1"
        ).glob(f"{arm_site_id}ceil10m{arm_subsite_id}*.nc"):
            # Example filename: sgpceil10mC1.b1.20231130.000008.custom.nc
            filename = path.name
            date_str = filename[16:24]
            try:
                date = pd.to_datetime(date_str, format="%Y%m%d")
                if date not in collected_dates:
                    collected_dates.add(date)
                    yield date, path
            except ValueError:
                # Invalid date string like 2024-13-99
                continue

        for path in (
            _arm_data_path() / "ceil" / f"{arm_site_id}ceil{arm_subsite_id}.b1"
        ).glob(f"{arm_site_id}ceil{arm_subsite_id}*.nc"):
            filename = path.name
            date_str = filename[13:21]
            try:
                date = pd.to_datetime(date_str, format="%Y%m%d")
                if date not in collected_dates:
                    collected_dates.add(date)
                    yield date, path
            except ValueError:
                continue


def get_arm_data_files(sites: DataFrame) -> Generator[ARMDataFile, None, None]:
    """Generate ARMDataFile objects for sites in DataFrame.

    Args:
        sites (pd.DataFrame): DataFrame with site metadata including arm_id column

    Yields:
        ARMDataFile: Data file wrapper for each available site-date combination
    """
    for site_id, arm_meta in sites.iterrows():
        assert isinstance(
            site_id, str
        )  # Theoretically could be any Hashable type
        arm_ids = arm_meta["arm_ids"]
        # We first iterate over available dates so we can later prioritize 10 m data.
        for date, data_file_path in collect_arm_data_files(arm_ids):
            try:
                data_file = ARMDataFile.from_source_data_file(
                    site_id=site_id,
                    site_name=arm_meta["humanReadableName"],
                    date=date,
                    data_path=data_file_path,
                )
            except DimensionalityError as e:
                warnings.warn(str(e), DimensionalityWarning)
                continue
            yield data_file
