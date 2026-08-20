import warnings
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from lcc_trend_analysis.observations.ceilometers import (
    CEILOMETERS,
)
from lcc_trend_analysis.observations.eprofile import (
    EprofileDataFile,
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


def _eprofile_data_path() -> Path:
    return get_data_paths().eprofile


def iter_available_eprofile_dates(
    eprofile_site_name: str, eprofile_country: str
) -> Generator[Timestamp, None, None]:
    """Iterate over available dates for an E-PROFILE site.

    Args:
        eprofile_site_name (str): E-PROFILE site name
        eprofile_country (str): Country code for the site

    Yields:
        Timestamp: Available observation dates for the site
    """
    collected_dates = set()
    for path in _eprofile_data_path().rglob(
        f"{eprofile_country}/{eprofile_site_name}/*"
    ):
        # Expecting path ./YYYY/MM/DD/site_id
        if len(path.parts) < 4:
            continue
        year, month, day = path.parts[-4:-1]
        try:
            date = pd.Timestamp(f"{year}-{month}-{day}")
            if date not in collected_dates:
                collected_dates.add(date)
                yield date
        except ValueError:
            # Invalid date string like 2024-13-99
            continue


def get_data_file_for_site_and_date(
    site_id: str, product: str, date: Timestamp
) -> Optional[Path]:
    path = (
        _eprofile_data_path()
        / date.strftime("%Y")
        / date.strftime("%m")
        / date.strftime("%d")
        / site_id
        / product
    )
    if path.exists():
        # Try to find data file with a preferred instrument order
        for instrument in CEILOMETERS:
            instrument_name = instrument.name.lower()
            file_paths = list(path.glob(pattern=f"*{instrument_name}*.nc"))
            if file_paths:
                # Return the last file found, assuming it is the most recent
                return file_paths[-1]
        # Try to find any .nc file if no specific instrument is found
        file_paths = list(path.glob(pattern="*.nc"))
        if len(file_paths) > 0:
            return file_paths[0]

    return None


def get_eprofile_data_product_files(
    sites: GeoDataFrame, product: str
) -> Generator[EprofileDataFile, None, None]:
    """Generate EprofileDataFile objects for sites and product.

    Args:
        sites (GeoDataFrame): Sites with eprofile_site_name and eprofile_country columns
        product (str): Data product name

    Yields:
        EprofileDataFile: Data file wrapper for each available site-date combination
    """
    for site_id, row in sites.iterrows():
        eprofile_site_name = row["eprofile_site_name"]
        eprofile_country = row["eprofile_country"]
        for date in iter_available_eprofile_dates(
            eprofile_site_name, eprofile_country
        ):
            data_file_path = get_data_file_for_site_and_date(
                eprofile_site_name, product, date
            )
            if data_file_path is None:
                continue
            try:
                data_file = EprofileDataFile(
                    date=date, site_id=site_id, data_path=data_file_path
                )
            except DimensionalityError as e:
                warnings.warn(str(e), DimensionalityWarning)
                continue
            yield data_file
