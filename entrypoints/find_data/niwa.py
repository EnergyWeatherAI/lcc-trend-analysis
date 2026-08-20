import re
import warnings
from pathlib import Path
from typing import Generator

import pandas as pd

from lcc_trend_analysis.observations.niwa import NIWADataFile
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.utils import (
    DimensionalityError,
    DimensionalityWarning,
)
from lcc_trend_analysis.type_aliases import GeoDataFrame, Timestamp


def niwa_data_path() -> Path:
    return get_data_paths().niwa


def collect_niwa_lauder_data_files() -> Generator[tuple[Timestamp, str, Path], None, None]:
    # Pattern to match cl61 filenames: lauder_YYYYMMDD.nc
    date_regex = re.compile(r"lauder_(\d{8}).nc$")
    for path in (niwa_data_path() / "cl61").rglob("*.nc"):
        match = date_regex.search(path.name)
        if match:
            date_str = match.group(1)
            try:
                date = pd.to_datetime(arg=date_str, format="%Y%m%d")
                yield date, "cl61", path
            except ValueError:
                continue
    
    date_regex = re.compile(r"LAUDER_CL31_(\d{8})_00.nc$")
    for path in (niwa_data_path() / "cl31").rglob("*.nc"):
        match = date_regex.search(path.name)
        if match:
            date_str = match.group(1)
            try:
                date = pd.to_datetime(arg=date_str, format="%Y%m%d")
                yield date, "cl31", path
            except ValueError:
                continue



def get_niwa_data_files(
    sites: GeoDataFrame,
) -> Generator[NIWADataFile, None, None]:
    site_id = "lauder"
    for date, instrument, data_file_path in collect_niwa_lauder_data_files():
        try:
            data_file = NIWADataFile.from_source_data_file(
                site_id=site_id,
                site_name=sites.loc[site_id, "humanReadableName"],
                date=date,
                data_path=data_file_path,
            )
            data_file.instrument = data_file.find_ceilometer_by_name(instrument)
        except DimensionalityError as e:
            warnings.warn(str(e), DimensionalityWarning)
            continue
        yield data_file
