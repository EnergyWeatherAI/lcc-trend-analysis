import logging
import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from entrypoints.find_data.fmi import get_fmi_data_files
from entrypoints.find_data.niwa import get_niwa_data_files
from lcc_trend_analysis.logging import (
    get_logger,
    setup_logging,
)
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.ceilometers import (
    CeilometerDataFile,
)
from lcc_trend_analysis.observations.utils import (
    get_ground_sites_gdf,
)
from lcc_trend_analysis.parallel_processing import (
    parallel_map,
)
from lcc_trend_analysis.type_aliases import (
    GeoDataFrame,
    Timestamp,
)

from .find_data.arm import get_arm_data_files
from .find_data.ceda import get_chilbolton_data_files
from .find_data.cloudnet import get_cloudnet_data_product_files
from .find_data.knmi import get_knmi_data_files
from .find_data.meteoswiss import get_payerne_data_files
from .find_data.metno import get_metno_data_files

logger: logging.Logger = get_logger(__name__)

load_dotenv()
DATA_PATHS = get_data_paths()

###
TIMEOUT = 600.0  # Per-task timeout to prevent hanging on corrupted files
TARGET_PATH_BASE: Path = DATA_PATHS.alc_level1b
N_JOBS: int = int(os.environ.get("N_JOBS", multiprocessing.cpu_count()))
START_TIME: Timestamp = pd.Timestamp("2000-01-01")
END_TIME: Timestamp = pd.Timestamp(pd.Timestamp.now())
###


@dataclass
class ALCFileMetadata:
    """Container for daily cloud cover processing results."""

    site_id: str
    instrument: str
    date: pd.Timestamp
    rel_path: Path
    source_data_path: str


def map_metadata_into_dataframe(
    file_metadata_results: list[ALCFileMetadata],
) -> pd.DataFrame:
    """Map list of ALCFileMetadata objects into pandas.DataFrame.

    Args:
        results (list[ALCFileMetadata]): List of ALCFileMetadata objects.
    Returns:
        pd.DataFrame: DataFrame containing metadata for processed files.
    """

    logger.info(
        f"Gathering metadata from {len(file_metadata_results)} daily files into a DataFrame."
    )

    df = pd.DataFrame(
        [
            {
                "site_id": result.site_id,
                "instrument": result.instrument,
                "date": result.date,
                "rel_path": str(result.rel_path),
                "source_data_path": result.source_data_path,
            }
            for result in file_metadata_results
        ]
    )

    return df


def preprocess_and_store_data_file(
    data_file: CeilometerDataFile,
    clobber: Optional[bool] = False,
) -> Optional[ALCFileMetadata]:
    """Process one daily ceilometer file to intermediate harmonized format and store it.

    Args:
        data_file (CeilometerDataFile): Ceilometer data file to process
        clobber (Optional[bool]): Whether to overwrite existing outputs

    Returns:
        ALCFileMetadata | None: Metadata for the processed file, or None if processing fails
    """
    
    if data_file.instrument is None:
        data_file.set_instrument()

    if data_file.instrument is None:
        logger.error(
            f"Failed to resolve instrument for source file {data_file.data_path}"
        )
        return None

    instrument = data_file.instrument
    
    target_path = (
        TARGET_PATH_BASE
        / data_file.site_id
        / data_file.date.strftime("%Y")
        / data_file.date.strftime("%m")
        / f"{data_file.date.strftime('%Y%m%d')}_{data_file.site_id}_{instrument.name.lower()}.nc"
    )
    if target_path.exists() and not clobber:
        logging.debug(
            f"Skipping {data_file.site_id} {instrument.name} data from {data_file.date} - already processed"
        )
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data_file.preprocess()
    except Exception as e:
        logger.error(
            f"Failed to preprocess data file {data_file.data_path} for {data_file.site_id} {instrument.name} on {data_file.date}: {e}"
        )
        return None

    if data_file.data is None:
        raise ValueError(
            f"Data file is empty or failed to load: {data_file.data_path}"
        )

    data_file.to_netcdf_product(path=target_path)

    return ALCFileMetadata(
        site_id=data_file.site_id,
        instrument=instrument.name.lower(),
        date=data_file.date,
        rel_path=target_path.relative_to(TARGET_PATH_BASE),
        source_data_path=str(data_file.data_path),
    )


def collect_and_preprocess_alc_source_data_files(
    sites: GeoDataFrame,
) -> None:
    """Collect and harmonise daily ALC data files from multiple sources into a single archive.

    Args:
        sites (GeoDataFrame): Metadata for ground sites
    """

    all_results = []

    processing_function = partial(
        preprocess_and_store_data_file,
        clobber=False,
    )

    # Step 1: ARM data
    logger.info("Collecting ARM data files...")
    arm_sites = sites.where(sites["arm_site"]).dropna(how="all")  # type: ignore
    results = parallel_map(
        func=processing_function,
        tasks=get_arm_data_files(arm_sites),
        n_jobs=N_JOBS,
        timeout=TIMEOUT,
    )
    all_results.extend([r for r in results if r is not None])
    logger.info(
        f"  Collected {len([r for r in results if r is not None])} data files from ARM"
    )

    # FMI data
    logger.info("Collecting FMI data files...")
    results = parallel_map(
        func=processing_function,
        tasks=get_fmi_data_files(
            sites.loc[
                [
                    "kenttarova",
                    "kumpula",
                    "uto",
                    "vehmasmaki",
                ]
            ]
        ),
        n_jobs=N_JOBS,
        timeout=TIMEOUT,
    )
    all_results.extend([r for r in results if r is not None])
    logger.info(
        f"  Collected {len([r for r in results if r is not None])} data files from FMI"
    )

    # Step 3: CEDA Chilbolton CT75k data
    if "chilbolton" in sites.index:
        logger.info("Collecting CEDA Chilbolton CT75k files...")
        results = parallel_map(
            func=processing_function,
            tasks=get_chilbolton_data_files(),
            n_jobs=N_JOBS,
            timeout=TIMEOUT,
        )
        all_results.extend([r for r in results if r is not None])
        logger.info(
            f"  Collected {len([r for r in results if r is not None])} data files from CEDA"
        )

    # Step 4: MeteoSwiss Payerne CHM15k data
    if "payerne" in sites.index:
        logger.info("Collecting MeteoSwiss Payerne CHM15k files...")
        results = parallel_map(
            func=processing_function,
            tasks=get_payerne_data_files(),
            n_jobs=N_JOBS,
            timeout=TIMEOUT,
        )
        all_results.extend([r for r in results if r is not None])
        logger.info(
            f"  Collected {len([r for r in results if r is not None])} data files from MeteoSwiss"
        )

    # MET Norway Oslo and Flesland CHM15k data
    logger.info("Collecting MET Norway CHM15k files...")
    results = parallel_map(
        func=processing_function,
        tasks=get_metno_data_files(sites.loc[["oslo", "flesland"]]),
        n_jobs=N_JOBS,
        timeout=TIMEOUT,
    )

    all_results.extend([r for r in results if r is not None])
    logger.info(
        f"  Collected {len([r for r in results if r is not None])} data files from MET Norway"
    )

    # KNMI Cabauw data
    if "cabauw" in sites.index:
        logger.info("Collecting KNMI files...")
        results = parallel_map(
            func=processing_function,
            tasks=get_knmi_data_files(sites.loc[["cabauw"]]),
            n_jobs=N_JOBS,
            timeout=TIMEOUT,
        )

        all_results.extend([r for r in results if r is not None])
        logger.info(
            f"  Collected {len([r for r in results if r is not None])} data files from KNMI"
        )
        
    # NIWA Lauder data
    if "lauder" in sites.index:
        logger.info("Collecting NIWA Lauder files...")
        results = parallel_map(
            func=processing_function,
            tasks=get_niwa_data_files(sites.loc[["lauder"]]),
            n_jobs=N_JOBS,
            timeout=TIMEOUT,
        )

        all_results.extend([r for r in results if r is not None])
        logger.info(
            f"  Collected {len([r for r in results if r is not None])} data files from NIWA"
        )
    
    # Fill the rest of the data with Cloudnet products where available
    logger.info("Collecting Cloudnet lidar products...")
    cloudnet_sites = sites.where(~sites["arm_site"]).dropna(how="all")  # type: ignore
    results = parallel_map(
        func=processing_function,
        tasks=get_cloudnet_data_product_files(cloudnet_sites, product="lidar"),
        n_jobs=N_JOBS,
        timeout=TIMEOUT,
    )
    all_results.extend([r for r in results if r is not None])
    logger.info(
        f"  Collected {len([r for r in results if r is not None])} data files from Cloudnet lidar products"
    )

    logger.info("")
    logger.info(f"Total collected: {len(all_results)} daily files")

    df_metadata = map_metadata_into_dataframe(all_results)
    metadata_path = DATA_PATHS.alc_level1b_source_metadata
    df_metadata.to_parquet(metadata_path, index=None)
    logger.info(f"Stored metadata DataFrame at: {metadata_path}")


def run():
    """Collect and preprocess ceilometer data files from various sources under a single directory tree."""
    logger.info("=" * 80)
    logger.info("Collect and preprocess ALC data files")
    logger.info("=" * 80)

    sites: GeoDataFrame = get_ground_sites_gdf()

    logger.info(f"Loaded metadata for {len(sites)} ground sites")

    collect_and_preprocess_alc_source_data_files(sites)

    logger.info("=" * 80)
    logger.info("Data collection completed successfully.")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging(logging.INFO)

    run()
