"""Process ceilometer observations to compute raw cloud cover time series.

Pipeline:
1. Discover data files from multiple sources (Cloudnet, ARM, CEDA, MeteoSwiss)
2. Process each daily file to compute native-sample 1D cloud-product variables
3. Aggregate results into a single dataset spanning all sites and times
4. Individual file failures don't stop the pipeline (fault-tolerant processing)
"""

import logging
import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import Any, Optional, Callable, cast

import pandas as pd
from dotenv import load_dotenv

from lcc_trend_analysis.logging import (
    get_logger,
    setup_logging,
)
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.ceilometers import (
    Level1bcCeilometerDataFile,
)
from lcc_trend_analysis.observations.utils import (
    get_ground_sites_gdf,
)
from lcc_trend_analysis.type_aliases import (
    GeoDataFrame,
)

from .create_alc_level2_cloud_products import (
    DailyCloudCoverArtifact,
    RAW_LEVEL2_FILENAME,
    process_alc_data,
    process_cloud_cover,
)

logger: logging.Logger = get_logger(__name__)

load_dotenv()

DATA_PATHS = get_data_paths()

###
TIMEOUT = 180.0  # Per-task timeout to prevent hanging on corrupted files
N_JOBS: int = int(os.environ.get("N_JOBS", multiprocessing.cpu_count()))
###


def process_cloud_cover_with_modified_calibration(
    data_file: Level1bcCeilometerDataFile,
    target_path_base: Path,
    calibration_modification: float,
    start_time: Optional[pd.Timestamp] = None,
    clobber: Optional[bool] = False,
) -> Optional[DailyCloudCoverArtifact]:
    """Process one daily ceilometer file with monkey-patched calibration.

    Args:
        data_file (Level1bcCeilometerDataFile): _description_
        target_path_base (Path): _description_
        clobber (Optional[bool], optional): _description_. Defaults to False.

    Returns:
        Optional[DailyResult]: _description_
    """
    
    level1bc_data_file = cast(Level1bcCeilometerDataFile, data_file)

    if start_time is not None and level1bc_data_file.date < start_time:
        logger.debug(
            f"Skipping file {level1bc_data_file.data_path} before start time {start_time}"
        )
        return None

    # Store a reference to the original calibration function
    original_pre_calibration_function: Callable = level1bc_data_file.pre_calibration

    # Monkey-patch the pre-calibration function by modifying the calibration factor by the very end
    def modified_calibration_function(*args, **kwargs) -> None:
        original_pre_calibration_function(*args, **kwargs)
        instrument = level1bc_data_file.instrument
        if instrument is None:
            raise ValueError(
                f"Missing instrument metadata for {level1bc_data_file.data_path}."
            )
        cast(Any, instrument).calibration_coefficient *= (
            calibration_modification
        )

    level1bc_data_file.pre_calibration = modified_calibration_function  # type: ignore

    daily_result = process_cloud_cover(
        data_file=cast(Any, level1bc_data_file),
        target_path_base=target_path_base,
        clobber=clobber,
    )

    return daily_result


def run():
    """Process cloud cover from ceilometer observations across multiple data sources."""
    logger.info("=" * 80)
    logger.info("ALC PRODUCT CALIBRATION SENSITIVITY TESTS")
    logger.info("=" * 80)

    sites: GeoDataFrame = get_ground_sites_gdf()
    
    # Sensitivity test only a small subset of sites
    sites_subset = ["chilbolton", "nsa", "sgp", "graciosa", "granada", "juelich", "oslo", "flesland", "palaiseau", "payerne"]
    sites = sites[sites.index.isin(sites_subset)]

    logger.info(f"Loaded metadata for {len(sites)} ground sites")

    logger.info("")
    
    modifications = {
        "0.8x": 0.80,
        "1.2x": 1.20,
    }
    
    logger.info("Using calibration modifications: " + ", ".join(modifications.keys()))
    
    for modification_label, modification in modifications.items():
        logger.info(f"Processing with {modification_label} calibration modification...")
        target_path_base: Path = (
            DATA_PATHS.alc_level2_calibration_sensitivity
            / modification_label
        )

        raw_product_with_calibration_modification_path: Path = (
            target_path_base / RAW_LEVEL2_FILENAME
        )

        logger.info("")

        processing_function_with_calibration_modification = partial(
            process_cloud_cover_with_modified_calibration,
            target_path_base=target_path_base,
            calibration_modification=modification,
            start_time=pd.Timestamp("2015-01-01"),
            clobber=False,
        )

        process_alc_data(
            processing_function_with_calibration_modification,
            sites,
            target_path_base=target_path_base,
            raw_product_nc_path=raw_product_with_calibration_modification_path,
        )
        logger.info(
            f"Saved raw calibration-sensitivity product to {raw_product_with_calibration_modification_path}"
        )

    logger.info("=" * 80)
    logger.info("Processing complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging(logging.INFO)

    run()
