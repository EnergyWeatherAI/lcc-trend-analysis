import re
import warnings
from pathlib import Path
from typing import Generator, Iterable, Optional, cast

import pandas as pd

from lcc_trend_analysis.observations.metno import (
    MetnoCHM15kDataFile,
)
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.utils import (
    DimensionalityError,
    DimensionalityWarning,
)
from lcc_trend_analysis.type_aliases import (
    GeoDataFrame,
    Timestamp,
)



def _metno_data_path() -> Path:
    return get_data_paths().metno


def get_data_file_for_date(site_id: str, date: Timestamp) -> Optional[Path]:
    dir_path = (
        _metno_data_path() / site_id / date.strftime("%Y") / date.strftime("%m")
    )

    if not dir_path.exists():
        return None

    pattern = f"{date.strftime('%Y%m%d')}*.nc"
    matching_files = list(dir_path.glob(pattern))

    if matching_files:
        return matching_files[0]

    return None


def collect_metno_data_files(site_id: str) -> Generator[tuple[Timestamp, Path], None, None]:
    # Pattern to match filenames: YYYYMMDD*.nc
    date_regex = re.compile(r"^(\d{8}).*\.nc$")
    collected_dates = set()
    for path in (_metno_data_path() / site_id).rglob("*.nc"):
        match = date_regex.search(path.name)
        if match:
            date_str = match.group(1)
            try:
                date = pd.to_datetime(arg=date_str, format="%Y%m%d")
                if date not in collected_dates:
                    collected_dates.add(date)
                    yield date, path
            except ValueError:
                continue


def get_metno_data_files(
    sites: GeoDataFrame,
) -> Generator[MetnoCHM15kDataFile, None, None]:
    for site_id_raw, site_meta in sites.iterrows():
        site_id_str = cast(str, str(site_id_raw))
        for date, data_file_path in collect_metno_data_files(site_id=site_id_str):
            if data_file_path is None:
                continue
            try:
                data_file = MetnoCHM15kDataFile.from_source_data_file(
                    site_id=site_id_str,
                    site_name=site_meta["humanReadableName"],
                    date=date,
                    data_path=data_file_path,
                )
                data_file.instrument = data_file.find_ceilometer_by_name("chm15k")
            except DimensionalityError as e:
                warnings.warn(str(e), DimensionalityWarning)
                continue
            yield data_file
