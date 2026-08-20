import logging
import multiprocessing
import os
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Optional

import pandas as pd
import xarray as xr
from dotenv import load_dotenv

from lcc_trend_analysis.algorithms.cloud_detection import (
    ALCCloudCoverEstimator,
)
from lcc_trend_analysis.logging import (
    get_logger,
    setup_logging,
)
from lcc_trend_analysis.observations.ceilometers import (
    Level1bcCeilometerDataFile,
)
from lcc_trend_analysis.observations.utils import (
    get_ground_sites_gdf,
    mask_spurious_days,
)
from lcc_trend_analysis.parallel_processing import (
    parallel_map,
)
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.products.alc_cloud_cover import (
    ALCCloudCoverProduct,
    extract_raw_time_series_data,
)
from lcc_trend_analysis.type_aliases import (
    GeoDataFrame,
)

logger: logging.Logger = get_logger(__name__)

load_dotenv()
DATA_PATHS = get_data_paths()

###
TIMEOUT = 180.0
RAW_SEGMENT_TIMEOUT = None
ALC_COLLECTION_PATH: Path = DATA_PATHS.alc_level1c
RAW_LEVEL2_FILENAME = DATA_PATHS.level2_clouds_raw("alc").name
RAW_STAGING_SUBDIR = "_raw_staging"
RAW_SEGMENT_FREQUENCY = "year"
KEEP_RAW_STAGING_FILES = False

TARGET_PATH_BASE: Path = DATA_PATHS.alc_level2_clouds
N_JOBS: int = int(os.environ.get("N_JOBS", multiprocessing.cpu_count()))
VALID_RAW_SEGMENT_FREQUENCIES = {"year", "month"}
###

if RAW_SEGMENT_FREQUENCY not in VALID_RAW_SEGMENT_FREQUENCIES:
    raise ValueError(
        "Unsupported ALC raw segment frequency "
        f"'{RAW_SEGMENT_FREQUENCY}'. Expected one of {sorted(VALID_RAW_SEGMENT_FREQUENCIES)}."
    )

STATE_VAR_THRESHOLDS: dict[str, float | int] = {
    "state_detector": 90.0,
    "state_laser": 80.0,
    "state_optics": 60.0,
#    "receiver_sens": 80.0,
#    "receiver_sensitivity": 80.0,
    "laser_energy": 88.0,
    "window_transmission": 65.0,
    "qc_background_light": 0,
    "qc_laser_pulse_energy": 0,
    "qc_laser_temperature": 0,
    "qc_window_transmission": 0
}

@dataclass(frozen=True)
class DailyCloudCoverArtifact:
    """Metadata for one daily level-2 cloud-cover file."""

    site_id: str
    date: pd.Timestamp
    instrument_name: str
    data_path: Path


@dataclass(frozen=True)
class CloudCoverSegmentTask:
    """Bounded concat task for one site/instrument/period segment."""

    site_id: str
    instrument_name: str
    period_label: str
    sort_key: tuple[int, int]
    daily_paths: tuple[Path, ...]
    segment_path: Path


@dataclass(frozen=True)
class CloudCoverSegmentArtifact:
    """Compiled segment metadata used for final assembly."""

    site_id: str
    instrument_name: str
    period_label: str
    sort_key: tuple[int, int]
    segment_path: Path


def _group_path(site_id: str, instrument_name: str) -> str:
    return f"{site_id}/{instrument_name}"


def _staging_root(target_path_base: Path) -> Path:
    return target_path_base / RAW_STAGING_SUBDIR


def _daily_cloud_cover_path(
    data_file: Level1bcCeilometerDataFile,
    target_path_base: Path,
    instrument_name: str,
) -> Path:
    return (
        target_path_base
        / data_file.site_id
        / data_file.date.strftime("%Y")
        / data_file.date.strftime("%m")
        / f"{data_file.date.strftime('%Y%m%d')}_{data_file.site_id}_{instrument_name}_cloud_cover.nc"
    )


def _load_raw_time_series_dataset(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as ds_daily:
        ds_raw = extract_raw_time_series_data(ds_daily)
        return ds_raw.load()


def _combine_time_series_datasets(
    datasets: list[xr.Dataset],
    *,
    site_id: str,
    instrument_name: str,
) -> xr.Dataset:
    if not datasets:
        raise ValueError(
            "Cannot combine an empty set of raw cloud-cover datasets for "
            f"site='{site_id}', instrument='{instrument_name}'."
        )

    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = xr.concat(
            datasets,
            dim="time",
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )

    ds = ds.sortby("time")
    ds = ds.drop_duplicates("time")
    ds.attrs.update(
        {
            "site_id": site_id,
            "instrument_name": instrument_name,
        }
    )
    return ds


def _segment_period(date: pd.Timestamp) -> tuple[str, tuple[int, int]]:
    if RAW_SEGMENT_FREQUENCY == "month":
        return date.strftime("%Y%m"), (date.year, date.month)
    return date.strftime("%Y"), (date.year, 0)


def _segment_path(
    staging_root: Path,
    site_id: str,
    instrument_name: str,
    period_label: str,
) -> Path:
    path = staging_root / site_id / instrument_name
    return path / f"{period_label}.nc"


def build_segment_tasks(
    daily_artifacts: list[DailyCloudCoverArtifact],
    target_path_base: Path,
) -> list[CloudCoverSegmentTask]:
    grouped: dict[
        tuple[str, str, str, tuple[int, int]],
        list[DailyCloudCoverArtifact],
    ] = {}

    for artifact in daily_artifacts:
        period_label, sort_key = _segment_period(artifact.date)
        key = (
            artifact.site_id,
            artifact.instrument_name,
            period_label,
            sort_key,
        )
        grouped.setdefault(key, []).append(artifact)

    staging_root = _staging_root(target_path_base)
    segment_tasks: list[CloudCoverSegmentTask] = []
    for (
        site_id,
        instrument_name,
        period_label,
        sort_key,
    ), artifacts in sorted(grouped.items()):
        artifacts.sort(key=lambda artifact: artifact.date)
        segment_tasks.append(
            CloudCoverSegmentTask(
                site_id=site_id,
                instrument_name=instrument_name,
                period_label=period_label,
                sort_key=sort_key,
                daily_paths=tuple(artifact.data_path for artifact in artifacts),
                segment_path=_segment_path(
                    staging_root,
                    site_id,
                    instrument_name,
                    period_label,
                ),
            )
        )

    return segment_tasks


def compile_raw_segment(
    segment_task: CloudCoverSegmentTask,
) -> Optional[CloudCoverSegmentArtifact]:
    if not segment_task.daily_paths:
        return None

    datasets = [
        _load_raw_time_series_dataset(daily_path)
        for daily_path in segment_task.daily_paths
    ]
    ds_segment = _combine_time_series_datasets(
        datasets,
        site_id=segment_task.site_id,
        instrument_name=segment_task.instrument_name,
    )
    ds_segment = mask_spurious_days(
        ds_segment,
        site_id=segment_task.site_id,
        instrument_name=segment_task.instrument_name,
    )

    segment_task.segment_path.parent.mkdir(parents=True, exist_ok=True)
    ds_segment.to_netcdf(
        segment_task.segment_path,
        mode="w",
        encoding= {
            "time": {
                "units": "seconds since 1970-01-01 00:00:00",
                "dtype": "float64",
            }
        },
        unlimited_dims=["time"],
    )

    return CloudCoverSegmentArtifact(
        site_id=segment_task.site_id,
        instrument_name=segment_task.instrument_name,
        period_label=segment_task.period_label,
        sort_key=segment_task.sort_key,
        segment_path=segment_task.segment_path,
    )


def assemble_raw_product(
    segment_artifacts: list[CloudCoverSegmentArtifact],
    raw_product_nc_path: Path,
) -> None:
    if not segment_artifacts:
        raise ValueError("No raw segment artifacts were produced for final assembly.")

    grouped_segments: dict[
        tuple[str, str],
        list[CloudCoverSegmentArtifact],
    ] = {}
    for segment_artifact in segment_artifacts:
        key = (
            segment_artifact.site_id,
            segment_artifact.instrument_name,
        )
        grouped_segments.setdefault(key, []).append(segment_artifact)

    raw_product_nc_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_product_nc_path.exists():
        raw_product_nc_path.unlink()

    first_write = True
    for site_id, instrument_name in sorted(grouped_segments):
        segment_group = grouped_segments[(site_id, instrument_name)]
        segment_group.sort(key=lambda segment_artifact: segment_artifact.sort_key)

        datasets = [
            _load_raw_time_series_dataset(segment_artifact.segment_path)
            for segment_artifact in segment_group
        ]
        ds_final = _combine_time_series_datasets(
            datasets,
            site_id=site_id,
            instrument_name=instrument_name,
        )

        ds_final.to_netcdf(
            raw_product_nc_path,
            mode="w" if first_write else "a",
            group=_group_path(site_id, instrument_name),
            engine="netcdf4",
            encoding= {
                "time": {
                    "units": "seconds since 1970-01-01 00:00:00",
                    "dtype": "float64",
                }
            },
            unlimited_dims=["time"],
        )
        first_write = False


def cleanup_staging_dir(staging_root: Path) -> None:
    if staging_root.exists():
        shutil.rmtree(staging_root)


def process_cloud_cover(
    data_file: Level1bcCeilometerDataFile,
    target_path_base: Path = TARGET_PATH_BASE,
    clobber: Optional[bool] = False,
) -> Optional[DailyCloudCoverArtifact]:
    """Process one daily ceilometer file and return the daily artifact metadata."""
    instrument_name = (
        data_file.instrument.name.lower()
        if data_file.instrument is not None
        else "unknown"
    )

    target_path = _daily_cloud_cover_path(
        data_file,
        target_path_base,
        instrument_name,
    )
    if target_path.exists() and not clobber:
        logging.debug(
            f"Skipping {data_file.site_id} {data_file.date} - already processed"
        )
        return DailyCloudCoverArtifact(
            site_id=data_file.site_id,
            date=data_file.date,
            instrument_name=instrument_name,
            data_path=target_path,
        )

    if data_file.data is None:
        raise ValueError(
            f"Data file is empty or failed to load: {data_file.data_path}"
        )

    cloud_cover_product: ALCCloudCoverProduct = ALCCloudCoverProduct.from_data_file(
        data_file,
        estimator=ALCCloudCoverEstimator(),
    )
    
    cloud_cover_product.to_netcdf(path=target_path)

    return DailyCloudCoverArtifact(
        site_id=data_file.site_id,
        date=data_file.date,
        instrument_name=instrument_name,
        data_path=target_path,
    )


def get_alc_data_collection_files(
    sites: GeoDataFrame,
) -> Generator[Level1bcCeilometerDataFile, None, None]:
    """Get metadata of all ceilometer data files in the preprocessed ALC data collection."""

    data_file_metadata_path = DATA_PATHS.alc_level1c_source_metadata
    df_metadata = pd.read_parquet(data_file_metadata_path)

    for _, row in df_metadata.iterrows():
        if any(
            (col.startswith("qc_") and row[col] > STATE_VAR_THRESHOLDS[col])
            or (not col.startswith("qc_") and row[col] < STATE_VAR_THRESHOLDS[col])
            for col in STATE_VAR_THRESHOLDS
            if col in row.index
        ):
            continue
        yield Level1bcCeilometerDataFile.from_level1bc_data_file(
            row["site_id"],
            sites.loc[row["site_id"], "humanReadableName"],
            row["instrument"],
            row["date"],
            data_path=ALC_COLLECTION_PATH / row["rel_path"],
        )


def process_alc_data(
    processing_function: Callable,
    sites: GeoDataFrame,
    target_path_base: Path,
    raw_product_nc_path: Path,
) -> Path:
    """Process ALC cloud-cover data into a staged raw concatenated dataset."""

    logger.info("Processing ALC cloud cover data from preprocessed data collection...")
    daily_artifacts = parallel_map(
        func=processing_function,
        tasks=get_alc_data_collection_files(sites),
        n_jobs=N_JOBS,
        timeout=TIMEOUT,
    )

    logger.info("")
    logger.info(f"Collected {len(daily_artifacts)} daily cloud-cover artifacts")

    if not daily_artifacts:
        raise ValueError("No daily cloud-cover artifacts were produced.")

    staging_root = _staging_root(target_path_base)
    cleanup_staging_dir(staging_root)

    segment_tasks = build_segment_tasks(daily_artifacts, target_path_base)
    logger.info(
        "Compiling raw cloud cover segments with %s granularity (%d segments)...",
        RAW_SEGMENT_FREQUENCY,
        len(segment_tasks),
    )
    segment_artifacts = parallel_map(
        func=compile_raw_segment,
        tasks=segment_tasks,
        n_jobs=N_JOBS,
        timeout=RAW_SEGMENT_TIMEOUT,
    )

    logger.info("Assembling final raw cloud cover dataset...")
    assemble_raw_product(segment_artifacts, raw_product_nc_path)

    if KEEP_RAW_STAGING_FILES:
        logger.info(f"Keeping raw segment staging files under {staging_root}")
    else:
        cleanup_staging_dir(staging_root)

    return raw_product_nc_path


def run() -> None:
    """Process cloud cover from ceilometer observations across multiple data sources."""
    logger.info("=" * 80)
    logger.info("CLOUD COVER FROM ALC OBSERVATIONS")
    logger.info("=" * 80)

    sites: GeoDataFrame = get_ground_sites_gdf()

    logger.info(f"Loaded metadata for {len(sites)} ground sites")

    raw_product_nc_path = TARGET_PATH_BASE / RAW_LEVEL2_FILENAME

    logger.info("")

    processing_function = partial(
        process_cloud_cover,
        target_path_base=TARGET_PATH_BASE,
        clobber=False,
    )

    process_alc_data(
        processing_function,
        sites,
        target_path_base=TARGET_PATH_BASE,
        raw_product_nc_path=raw_product_nc_path,
    )

    logger.info(f"Saved raw level-2 product to {raw_product_nc_path}")
    logger.info("=" * 80)
    logger.info("Processing complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging(logging.INFO)

    run()
