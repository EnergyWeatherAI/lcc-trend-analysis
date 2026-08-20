import re
import warnings
from pathlib import Path
from typing import Generator

import pandas as pd

from lcc_trend_analysis.observations.meteoswiss import (
    MeteoSwissCHM15kDataFile,
)
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.utils import (
    DimensionalityError,
    DimensionalityWarning,
)
from lcc_trend_analysis.type_aliases import Timestamp


def _meteoswiss_data_path() -> Path:
    return get_data_paths().meteo_swiss_lidar_chm15k


def collect_meteoswiss_payerne_data_files() -> Generator[
    tuple[Timestamp, Path], None, None
]:
    # Pattern to match filenames: YYYYMMDD*.nc
    date_regex = re.compile(r"^(\d{8}).*\.nc$")
    collected_dates = set()
    for path in _meteoswiss_data_path().rglob("*.nc"):
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


def get_payerne_data_files() -> Generator[
    MeteoSwissCHM15kDataFile, None, None
]:
    for date, data_file_path in collect_meteoswiss_payerne_data_files():
        if data_file_path is None:
            continue
        try:
            data_file = MeteoSwissCHM15kDataFile.from_source_data_file(
                site_id="payerne",
                site_name="Payerne",
                date=date,
                data_path=data_file_path,
            )
            data_file.instrument = data_file.find_ceilometer_by_name("chm15k")
        except DimensionalityError as e:
            warnings.warn(str(e), DimensionalityWarning)
            continue
        yield data_file
