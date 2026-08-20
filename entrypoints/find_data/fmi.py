import warnings
from pathlib import Path
from typing import Generator

import pandas as pd

from lcc_trend_analysis.observations.fmi import (
    FMIDataFile,
)
from lcc_trend_analysis.paths import get_data_paths

from lcc_trend_analysis.type_aliases import (
    GeoDataFrame,
    Timestamp,
)


def _fmi_data_path() -> Path:
    return get_data_paths().fmi


def iter_available_instruments(site_id: str) -> Generator[str, None, None]:
    """Iterate over available instruments for a Cloudnet site.

    Args:
        site_id (str): Cloudnet site identifier
    """
    site_path: Path = _fmi_data_path() / site_id
    if not site_path.exists():
        return

    for instrument_path in site_path.iterdir():
        if instrument_path.is_dir():
            yield instrument_path.name


def collect_fmi_data_files(
    site_id: str, instrument_key: str
) -> Generator[tuple[Timestamp, Path], None, None]:
    """Iterate over available dates for a Cloudnet site.

    Args:
        site_id (str): Cloudnet site identifier
        instrument_key (str): Instrument identifier
    Yields:
        tuple[Timestamp, Path]: Available observation dates and corresponding file paths for the given site and instrument
    """
    collected_dates = set()
    for path in (_fmi_data_path() / site_id / instrument_key).rglob("*.nc"):
        # Expecting path ./YYYY/YYYYMMDD_site_instrument_id.nc
        year = int(path.name[:4])
        month = int(path.name[4:6])
        day = int(path.name[6:8])
        try:
            date = pd.Timestamp(f"{year}-{month}-{day}")
            if date not in collected_dates:
                collected_dates.add(date)
                yield date, path
        except ValueError:
            # Invalid date string like 2024-13-99
            continue


def get_fmi_data_files(
    sites: GeoDataFrame,
) -> Generator[FMIDataFile, None, None]:
    """Generate CloudnetDataFile objects for sites and instruments.

    Args:
        sites (GeoDataFrame): Sites to process

    Yields:
        FMIDataFile: Data file wrapper for each available site-date combination
    """
    for site_id, site_meta in sites.iterrows():
        site_id = str(site_id)
        for instrument_key in iter_available_instruments(site_id):
            for date, data_file_path in collect_fmi_data_files(
                site_id, instrument_key
            ):
                if data_file_path is None:
                    continue
                try:
                    data_file = FMIDataFile.from_source_data_file(
                        site_id=site_id,
                        site_name=site_meta["humanReadableName"],
                        date=date,
                        data_path=data_file_path,
                    )
                    data_file.instrument = data_file.find_ceilometer_by_name(instrument_key)
                except Exception as e:
                    warnings.warn(
                        f"Failed to create FMIDataFile for {data_file_path}: {e}",
                        category=UserWarning,
                    )
                    continue
                yield data_file
