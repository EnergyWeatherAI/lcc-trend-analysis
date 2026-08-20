import warnings
from pathlib import Path
from typing import Generator

from lcc_trend_analysis.observations.cloudsat import (
    CloudSat2BDataFile,
    CloudSatDataFile,
)
from lcc_trend_analysis.paths import get_data_paths


def _cloudsat_data_path() -> Path:
    return get_data_paths().cloudsat


def get_cloudsat_product_paths(product: str) -> list[Path]:
    """Return CloudSat raw-product directories for a logical product name.

    Raw CloudSat directories are versioned, for example
    `2B-CLDCLASS-LIDAR.P1_R05` and `ECMWF-AUX.P1_R05`.
    """
    base_path = _cloudsat_data_path()
    return sorted(
        path for path in base_path.glob(f"{product}*") if path.is_dir()
    )


def get_cloudsat_data_files(
    product: str,
) -> Generator[CloudSatDataFile, None, None]:
    """Get CloudSat data files for a specific product.

    Args:
        product (str): CloudSat product name (e.g. "2B-GEOPROF")

    Yields:
        CloudSatDataFile: CloudSat data file object for each file found
    """
    product_paths = get_cloudsat_product_paths(product)
    if not product_paths:
        warnings.warn(
            f"No CloudSat product directories found for {product} under {_cloudsat_data_path()}"
        )
        return

    for product_path in product_paths:
        for file_path in sorted(product_path.rglob(f"*{product}*.hdf")):
            try:
                data_file = CloudSat2BDataFile(
                    data_path=file_path, product=product
                )
                yield data_file
            except Exception as e:
                warnings.warn(
                    f"Failed to create CloudSat2BDataFile for {file_path}: {e}"
                )
                continue
