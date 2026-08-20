import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

from lcc_trend_analysis.logging import get_logger, setup_logging
from lcc_trend_analysis.observations.ceres import (
    CERES_SYN1DEG_DAY_PRODUCT,
)
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.type_aliases import Dataset

logger: logging.Logger = get_logger(__name__)

load_dotenv()

DATA_PATHS = get_data_paths()

CERES_PRODUCT = "SYN1deg-1Hour"
TARGET_PATH_BASE: Path = DATA_PATHS.ceres_level2_clouds(CERES_PRODUCT)
RAW_LEVEL2_PATH: Path = DATA_PATHS.ceres_level2_clouds_raw(CERES_PRODUCT)

# CERES SYN1deg cloud_layer coordinate: 1=High, 2=UpperMid, 3=LowerMid, 4=Low, 5=Total
LOW_CLOUD_LAYER = 4
TOTAL_CLOUD_LAYER = 5
CERES_CLOUD_COVER_PREFIXES = ("obs", "adj")


def _load_level1_metadata() -> pd.DataFrame:
    metadata_path = DATA_PATHS.ceres_level1_metadata(CERES_PRODUCT)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"CERES level1 metadata not found at {metadata_path}. "
            "Run create_ceres_level1_products.py first."
        )
    metadata = pd.read_parquet(metadata_path)
    if metadata.empty:
        return metadata
    metadata["time"] = pd.to_datetime(metadata["time"])
    return metadata.sort_values("time").reset_index(drop=True)


def _compute_cloud_covers(ds: Dataset) -> Dataset:
    """Compute low cloud cover and non-obscured low cloud cover for each of the
    obs_/adj_ CERES cloud-cover variants.

    low_cloud_cover_X = X_cloud_cover at cloud_layer=4 (Low, 700 mb-Surface)
    low_cloud_cover_non_obscured_X = low_cloud_cover_X / (1 - total_cloud_cover_X + low_cloud_cover_X),
                                     clipped to [0, 1]
    """
    result = ds.drop_dims("cloud_layer")

    for prefix in CERES_CLOUD_COVER_PREFIXES:
        source_var = f"{prefix}_cloud_cover"
        if source_var not in ds.data_vars:
            continue

        cloud_cover = ds[source_var]
        low_cloud_cover = cloud_cover.sel(cloud_layer=LOW_CLOUD_LAYER, drop=True).astype(
            "float32"
        )
        total_cloud_cover = cloud_cover.sel(cloud_layer=TOTAL_CLOUD_LAYER, drop=True).astype("float32")

        total_cloud_cover.attrs = {
            "long_name": f"{prefix.capitalize()} total cloud cover",
            "units": "1",
            "comment": "CERES SYN1deg cloud cover at cloud_layer=5 (Total).",
        }
        low_cloud_cover.attrs = {
            "long_name": f"{prefix.capitalize()} low cloud cover",
            "units": "1",
            "comment": "CERES SYN1deg cloud cover at cloud_layer=4 (Low, 700 mb-Surface).",
        }
        denominator = 1.0 - total_cloud_cover + low_cloud_cover
        low_cloud_cover_non_obscured = xr.where(
            denominator > 0,
            low_cloud_cover / denominator,
            float("nan"),
        ).astype("float32")
        low_cloud_cover_non_obscured = low_cloud_cover_non_obscured.clip(
            min=0.0, max=1.0
        )
        low_cloud_cover_non_obscured.attrs = {
            "long_name": f"{prefix.capitalize()} non-obscured low cloud cover",
            "units": "1",
            "comment": (
                "Computed as low_cloud_cover / (1 - total_cloud_cover + low_cloud_cover), "
                "clipped to [0, 1]."
            ),
        }

        result = result.assign(
            {
                f"low_cloud_cover_{prefix}": low_cloud_cover,
                f"total_cloud_cover_{prefix}": total_cloud_cover,
                f"low_cloud_cover_non_obscured_{prefix}": low_cloud_cover_non_obscured,
            }
        )

    return result


def _concatenate_level1_datasets(level1_paths: list[Path]) -> Dataset:
    """Load, concatenate, and validate CERES level1 time series.

    A SYN1deg-Day file contributes one timestamp; a SYN1deg-1Hour file
    contributes 24 midpoint timestamps. Duplicate timestamps indicate
    conflicting source products and are rejected rather than silently merged.
    """
    datasets: list[Dataset] = []
    for index, level1_path in enumerate(level1_paths, start=1):
        logger.info("Loading file %d/%d", index, len(level1_paths))
        with xr.open_dataset(level1_path, engine="netcdf4") as ds:
            datasets.append(ds.load())

    if not datasets:
        raise ValueError("No CERES level1 datasets supplied")

    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    ).sortby("time")
    if combined.indexes["time"].has_duplicates:
        duplicate_times = combined.indexes["time"][combined.indexes["time"].duplicated()]
        raise ValueError(
            "Duplicate CERES level1 timestamps encountered: "
            f"{duplicate_times.unique().tolist()!r}"
        )
    return combined


def run() -> None:
    logger.info("=" * 80)
    logger.info("CERES LEVEL2 CLOUD FRACTIONS")
    logger.info("=" * 80)

    metadata = _load_level1_metadata()
    if metadata.empty:
        logger.warning("CERES level1 metadata is empty; nothing to process.")
        return

    level1_base_path = DATA_PATHS.ceres_level1_product(CERES_PRODUCT)
    level1_paths = sorted(
        {level1_base_path / str(rel_path) for rel_path in metadata["rel_path"]}
    )
    logger.info("Loading %d CERES level1 files", len(level1_paths))
    
    ds = _concatenate_level1_datasets(level1_paths)
    ds_metrics = _compute_cloud_covers(ds)
    ds_metrics = ds_metrics.load()
    for var_name in ds_metrics.data_vars:
        if ds_metrics[var_name].dtype == np.float64:
            ds_metrics[var_name] = ds_metrics[var_name].astype(np.float32)

    TARGET_PATH_BASE.mkdir(parents=True, exist_ok=True)
    ds_metrics.to_netcdf(
        RAW_LEVEL2_PATH,
        mode="w",
        format="NETCDF4",
        engine="netcdf4",
    )
    logger.info(
        "Saved CERES raw level2 product to %s",
        RAW_LEVEL2_PATH,
    )
    logger.info("=" * 80)
    logger.info("CERES level2 processing complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging(logging.INFO)
    run()
