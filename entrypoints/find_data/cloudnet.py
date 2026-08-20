import warnings
from pathlib import Path
from typing import Generator, cast

import pandas as pd

from lcc_trend_analysis.observations.cloudnet import (
    CloudnetDataFile,
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


def _cloudnet_data_path() -> Path:
    return get_data_paths().cloudnet


def get_data_files_for_site_and_date(
    site_id: str, product: str, date: Timestamp
) -> Generator[Path, None, None]:
    """Get Cloudnet data file path for a specific site, product, and date.

    Args:
        site_id (str): Cloudnet site identifier
        product (str): Cloudnet data product name
        date (Timestamp): Date of observations

    Returns:
        Generator[Path, None, None]: Generator of paths to data files
    """
    path = (
        _cloudnet_data_path()
        / product
        / site_id
        / date.strftime("%Y")
        / date.strftime("%m")
    )
    if path.exists():
        # Try to find data file with a preferred instrument order
        yield from path.glob(
            pattern=f"{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}*.nc"
        )

    return None


def collect_cloudnet_product_files(
    site_id: str, product: str
) -> Generator[tuple[Timestamp, Path], None, None]:
    """Iterate over available dates for a Cloudnet site.

    Args:
        site_id (str): Cloudnet site identifier

    Yields:
        tuple[Timestamp, Path]: Available observation dates and corresponding file paths for the site
    """
    for path in (_cloudnet_data_path() / product / site_id).rglob("*.nc"):
        # Expecting path ./product/site_id/YYYY/MM/YYYYMMDD_site_instrument_id.nc
        year = int(path.name[:4])
        month = int(path.name[4:6])
        day = int(path.name[6:8])
        try:
            date = pd.Timestamp(f"{year}-{month}-{day}")
            yield date, path
        except ValueError:
            # Invalid date string like 2024-13-99
            continue


def get_cloudnet_data_product_files(
    sites: GeoDataFrame, product: str
) -> Generator[CloudnetDataFile, None, None]:
    """Generate CloudnetDataFile objects for sites and product.

    Args:
        sites (Iterable[str]): Site identifiers to process
        product (str): Cloudnet data product name

    Yields:
        CloudnetDataFile: Data file wrapper for each available site-date combination
    """
    for site_id, site_meta in sites.iterrows():
        site_id_str = cast(str, str(site_id))
        for date, data_file_path in collect_cloudnet_product_files(
            site_id_str, product
        ):
            if data_file_path is None:
                continue
            try:
                data_file = CloudnetDataFile.from_source_data_file(
                    site_id=site_id_str,
                    site_name=site_meta["humanReadableName"],
                    date=date,
                    data_path=data_file_path,
                )
            except DimensionalityError as e:
                warnings.warn(str(e), DimensionalityWarning)
                continue
            yield data_file
