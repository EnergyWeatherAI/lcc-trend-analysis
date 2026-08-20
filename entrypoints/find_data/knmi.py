import re
import warnings
from pathlib import Path
from typing import Generator

import pandas as pd

from lcc_trend_analysis.observations.knmi import KNMIDataFile
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.utils import (
    DimensionalityError,
    DimensionalityWarning,
)
from lcc_trend_analysis.type_aliases import (
    GeoDataFrame,
    Timestamp,
)



def _knmi_data_path() -> Path:
    return get_data_paths().knmi


def collect_knmi_data_files(
    wmo_id: str,
) -> Generator[tuple[Timestamp, str, Path], None, None]:
    # Pattern to match CHM15K filenames: ceilonet_chm15k_*_wmo_id_YYYYMMDD.nc
    date_regex = re.compile(rf"ceilonet_chm15k_.*_{wmo_id}_A(\d{{8}})\.nc$")
    for path in (_knmi_data_path() / "ceilonet_chm15k").rglob("*.nc"):
        match = date_regex.search(path.name)
        if match:
            date_str = match.group(1)
            try:
                date = pd.to_datetime(arg=date_str, format="%Y%m%d")
                yield date, "chm15k", path
            except ValueError:
                continue

    if wmo_id == "06348":  # Data available only for Cabauw
        date_regex = re.compile(r"(\d{8})")
        for path in (_knmi_data_path() / "cabauw_chm15k").rglob("*.nc"):
            match = date_regex.search(path.name)
            if match:
                date_str = match.group(1)
                try:
                    date = pd.to_datetime(arg=date_str, format="%Y%m%d")
                    yield date, "chm15k", path
                except ValueError:
                    continue

        # Pattern to match CT75k filenames: cesar_ct75ceilometer_*YYYYMMDD.nc
        date_regex = re.compile(r"cesar_ct75ceilometer_.*_(\d{8})\.nc$")
        for path in (_knmi_data_path() / "cabauw_ct75k").rglob("*.nc"):
            match = date_regex.search(path.name)
            if match:
                date_str = match.group(1)
                try:
                    date = pd.to_datetime(arg=date_str, format="%Y%m%d")
                    yield date, "ct75k", path
                except ValueError:
                    continue

        # Pattern to match LD40 filenames: YYYYMMDD_cabauw_ld40.nc
        date_regex = re.compile(r"^(\d{8})_cabauw_ld40\.nc$")
        for path in (_knmi_data_path() / "cabauw_ld40").rglob("*.nc"):
            match = date_regex.search(path.name)
            if match:
                date_str = match.group(1)
                try:
                    date = pd.to_datetime(arg=date_str, format="%Y%m%d")
                    yield date, "ld40", path
                except ValueError:
                    continue


def get_knmi_data_files(
    sites: GeoDataFrame,
) -> Generator[KNMIDataFile, None, None]:
    knmi_station_wmo_ids = {
        "cabauw": "06348",
    }
    for site_id, site_meta in sites.iterrows():
        site_id = str(site_id)
        for date, instrument, data_file_path in collect_knmi_data_files(
            wmo_id=knmi_station_wmo_ids[site_id]
        ):
            try:
                data_file = KNMIDataFile.from_source_data_file(
                    site_id=site_id,
                    site_name=site_meta["humanReadableName"],
                    date=date,
                    data_path=data_file_path,
                )
                data_file.instrument = data_file.find_ceilometer_by_name(instrument)
            except DimensionalityError as e:
                warnings.warn(str(e), DimensionalityWarning)
                continue
            yield data_file
