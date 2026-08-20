import logging
import multiprocessing
import os
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster
from dotenv import load_dotenv
from xarray.core.treenode import group_subtrees

from lcc_trend_analysis.logging import (
    get_logger,
    setup_logging,
)
from lcc_trend_analysis.observations.ceilometers import (
    CEILOMETERS,
)
from lcc_trend_analysis.observations.ceres import (
    CERES_SYN1DEG_DAY_PRODUCT,
    is_ceres_hourly_product,
)
from lcc_trend_analysis.parallel_processing import parallel_map
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.utils import (
    compute_solar_zenith_angle_for_site,
    compute_solar_zenith_angle_for_sites,
    get_ground_sites_gdf,
    get_site_coordinate_lookup,
)

logger: logging.Logger = get_logger(__name__)


load_dotenv()

DATA_PATHS = get_data_paths()
N_JOBS: int = int(os.environ.get("N_JOBS", multiprocessing.cpu_count()))
SITE_TASK_TIMEOUT = float(os.environ.get("LEVEL3_SITE_TIMEOUT", "0")) or None

### SETUP PARAMETERS ###
DATASET_TO_PROCESS = "ceres"  # "alc" | "era5" | "ceres"

# CERES SYN1deg-Day is a daily composite (one sample per site per day), so a
# SZA-based daytime/nighttime split at a single daily timestamp is not
# meaningful. SYN1deg-1Hour supplies midpoint timestamps and uses the normal
# SZA split automatically.
CERES_PRODUCT = "SYN1deg-1Hour"
CERES_LEVEL2_IS_DAILY_PRODUCT = not is_ceres_hourly_product(CERES_PRODUCT)


def _is_alc_dataset(dataset_stem: str) -> bool:
    return dataset_stem.startswith("alc")


def _is_era5_dataset(dataset_stem: str) -> bool:
    return dataset_stem.startswith("era5")


def _is_ceres_dataset(dataset_stem: str) -> bool:
    return dataset_stem.startswith("ceres")


def _supports_day_night_split(
    dataset_to_process: str, ceres_is_daily_product: bool
) -> bool:
    """Whether the daytime/nighttime SZA-based split should be computed.

    CERES SYN1deg-Day level2 output has a single sample per site per day, so
    the split is skipped while `ceres_is_daily_product` is True. All other
    datasets (and future sub-daily CERES products) support the split.
    """
    if dataset_to_process == "ceres" and ceres_is_daily_product:
        return False
    return True


SUPPORTS_DAY_NIGHT_SPLIT = _supports_day_night_split(
    DATASET_TO_PROCESS, CERES_LEVEL2_IS_DAILY_PRODUCT
)

# CERES level2 QC-only intermediate variables that should not be carried
# through to level3 aggregated outputs.
CERES_LEVEL3_EXCLUDED_VARIABLES = (
    "total_cloud_cover_obs",
    "total_cloud_cover_adj",
    "upper_level_cloud_cover_obs",
    "upper_level_cloud_cover_adj",
)


def _level2_input_filename(dataset_stem: str) -> str:
    if _is_era5_dataset(dataset_stem):
        return f"{dataset_stem}_hourly.nc"
    if (
        _is_alc_dataset(dataset_stem)
        or _is_ceres_dataset(dataset_stem)
    ):
        return f"{dataset_stem}_raw.nc"
    raise ValueError(f"Unsupported cloud dataset stem: {dataset_stem}")


INPUT_DATA_PATH: Path = (
    DATA_PATHS.ceres_level2_clouds(CERES_PRODUCT)
    if DATASET_TO_PROCESS == "ceres"
    else DATA_PATHS.level2_clouds(DATASET_TO_PROCESS)
)
BASE_DATA_PATH: Path = (
    DATA_PATHS.ceres_level3_clouds(CERES_PRODUCT)
    if DATASET_TO_PROCESS == "ceres"
    else DATA_PATHS.level3_clouds(DATASET_TO_PROCESS)
)
LEVEL2_DATASET_STEM = f"{DATASET_TO_PROCESS}_level2_clouds"
LEVEL3_DATASET_STEM = f"{DATASET_TO_PROCESS}_level3_clouds"

AGGREGATED_FLAG_VARIABLE_METADATA: dict[str, dict[str, str]] = {
    "cloud_flag": {
        "name": "cloud_cover",
        "long_name": "Cloud cover fraction",
        "comment": "Fraction of samples with a cloud layer in the profile.",
    },
    "low_cloud_flag": {
        "name": "low_cloud_cover",
        "long_name": "Low cloud cover fraction",
        "comment": "Fraction of samples with a cloud layer below 2000 m a.g.l.",
    },
}

# Minimum number of valid daily samples required to compute a daily cloud cover value for a site
ALC_MIN_DAILY_SAMPLES = 720

# Minimum number of valid daily samples required for an individual ALC instrument
# to participate in shared site climatology construction.
ALC_MIN_INSTRUMENT_VALID_DAYS = 365

# Days to include in a running mean when computing the mean climatology.
N_RUNNING_CLIMATOLOGY_DAYS = 31
RUNNING_CLIMATOLOGY_WINSOR_LOWER_QUANTILE = 0.02
RUNNING_CLIMATOLOGY_WINSOR_UPPER_QUANTILE = 0.98

CONDITION_SUFFIX_MAP: dict[str, str] = {
    "all": "",
    "daytime": "_daytime",
    "nighttime": "_nighttime",
}

SHARED_CLIMATOLOGY_GROUP = "shared_climatology"
WEEKLY_RESAMPLE_FREQ = "W-MON"

# Minimum fraction of available data in an aggregation period
MIN_DATA_FRAC = 0.5
MIN_DATA_FRAC_RUNNING = 0.5  # For running means

# Minimum number of valid daily samples required to process a site
MIN_VALID_DAYS = 5 * 365

# Start year for the climatology
CLIMATOLOGY_START_YEAR = 2002
CLIMATOLOGY_END_YEAR = 2025
###


@dataclass(frozen=True)
class SiteLeafSpec:
    group_path: str
    instrument_id: str


@dataclass(frozen=True)
class SiteProcessingTask:
    dataset_path: Path
    output_dir: Path
    site_id: str
    site_coordinate_lookup: dict[str, tuple[float, float]]
    leaf_specs: tuple[SiteLeafSpec, ...]


def _shared_climatology_group(site_id: str) -> str:
    return f"{site_id}/{SHARED_CLIMATOLOGY_GROUP}"


@xr.register_dataarray_accessor("dt_noleap")
class _NoLeapDTAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj  # this is the DataArray

    @property
    def dayofyear(self):
        """
        Return a no-leap day-of-year (1–365) as a DataArray for groupby.
        """
        t = self._obj
        is_leap = t.dt.is_leap_year
        after_feb28 = t.dt.month > 2
        return (t.dt.dayofyear - (is_leap & after_feb28)).astype("int32")


def mean_min_frac_along_dim(da, p, dim="time"):
    """Compute the mean along specified dimension with data availability threshold.

    Returns mean only if at least p fraction of data is non-null, otherwise NaN.
    This ensures temporal aggregations represent sufficient observational sampling.

    Args:
        da (DataArray | Dataset): The input data array
        p (float): The fraction of valid data required to compute the mean
        dim (str): The dimension along which to compute the mean

    Returns:
        DataArray: The data reduced along the given dimension, or NaN if the fraction of valid data is not met
    """
    if DATASET_TO_PROCESS == "alc":
        aggregate = da.mean(dim=dim).where(
            da.notnull().sum(dim=dim) >= len(da[dim]) * p
        )
    else:
        aggregate = da.mean(dim=dim)
    return aggregate


def mean_min_frac_on_groupby(gb, p):
    """Compute the mean of the data array on a groupby object if a certain fraction of the data is available.

    Args:
        gb (DataArray | Dataset): The input data array
        p (float): The fraction of valid data required to compute the mean

    Returns:
        DataArray: The reduced groupby object, or NaN if the fraction of valid data is not met
    """
    aggregate = gb.where(gb.notnull().sum() >= len(gb) * p)
    return aggregate


def compute_anomaly_from_climatology(
    ds_daily: xr.Dataset,
    climatology_ds: xr.Dataset,
    suffix: str = "",
) -> xr.Dataset:
    """Compute anomaly by subtracting a provided climatology dataset."""
    doys = ds_daily.time.dt_noleap.dayofyear

    anomaly_ds = xr.Dataset()
    for var_name in ds_daily.data_vars:
        clim_aligned = climatology_ds[var_name].sel(doy=doys)
        anomaly_ds[var_name] = ds_daily[var_name] - clim_aligned
        anomaly_ds[var_name].attrs.update(
            {
                "long_name": f"{var_name} - anomaly from day-of-year {N_RUNNING_CLIMATOLOGY_DAYS}-day climatology{suffix}",
                "method": "observation - climatology",
            }
        )

    return anomaly_ds


def _reference_period_slice() -> slice:
    return slice(
        pd.Timestamp(f"{CLIMATOLOGY_START_YEAR}-01-01"),
        pd.Timestamp(f"{CLIMATOLOGY_END_YEAR}-12-31"),
    )


def _select_reference_period(ds_daily: xr.Dataset) -> xr.Dataset:
    ds_reference = ds_daily.sel(time=_reference_period_slice())
    if ds_reference.sizes.get("time", 0) == 0:
        raise ValueError(
            "No data available in climatology reference period "
            f"{CLIMATOLOGY_START_YEAR}-{CLIMATOLOGY_END_YEAR}."
        )
    return ds_reference


def _concatenate_time_series_datasets(
    datasets: list[xr.Dataset],
    attrs: Optional[dict[str, Any]] = None,
) -> xr.Dataset:
    valid_datasets = [ds for ds in datasets if ds.sizes.get("time", 0) > 0]
    if not valid_datasets:
        raise ValueError(
            "No datasets with time samples available for concatenation."
        )

    combined = xr.concat(
        valid_datasets,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    ).sortby("time")
    if attrs:
        combined = combined.copy(deep=False).assign_attrs(attrs)
    return combined


def _baseline_from_reference_period(ds_daily: xr.Dataset) -> xr.Dataset:
    return _select_reference_period(ds_daily).mean(dim="time")


def _center_climatology_on_zero(climatology_ds: xr.Dataset) -> xr.Dataset:
    return climatology_ds - climatology_ds.mean(dim="doy")


def _summary_climatologies_from_reference_period(
    ds_daily: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    ds_reference = _select_reference_period(ds_daily)
    mean_climatology = ds_reference.mean(dim="time")
    monthly_climatology = ds_reference.groupby("time.month").mean(dim="time")
    seasonal_climatology = ds_reference.groupby("time.season").mean(dim="time")
    return mean_climatology, monthly_climatology, seasonal_climatology


def _winsorized_mean_over_dim(
    ds: xr.Dataset,
    dim: str,
    lower_quantile: float,
    upper_quantile: float,
) -> xr.Dataset:
    if not 0.0 <= lower_quantile <= upper_quantile <= 1.0:
        raise ValueError(
            "Winsorization quantiles must satisfy 0 <= lower <= upper <= 1."
        )

    quantiles = ds.quantile(
        [lower_quantile, upper_quantile],
        dim=dim,
        skipna=True,
    )

    lower = quantiles.sel(quantile=lower_quantile, drop=True)
    upper = quantiles.sel(quantile=upper_quantile, drop=True)
    winsorized_mean = ds.clip(min=lower, max=upper).mean(
        dim=dim,
        skipna=True,
    )

    for var_name in winsorized_mean.data_vars:
        winsorized_mean[var_name].attrs = ds[var_name].attrs.copy()

    return winsorized_mean


def _compute_doy_running_climatology_dataset(
    ds_daily: xr.Dataset,
    n_days: int,
) -> xr.Dataset:
    if n_days < 1 or n_days > 365:
        raise ValueError("n_days must be in the range [1, 365].")

    # Robustifying step, run 5-day median filter before computing the running mean climatology
    # to reduce the influence of outliers.
    #ds_daily = ds_daily.rolling(time=5, center=True, min_periods=5).median()

    ds_daily = ds_daily.assign_coords(doy=ds_daily.time.dt_noleap.dayofyear)
    ds_reference = _select_reference_period(ds_daily)

    doys = ds_reference.time.dt_noleap.dayofyear.astype("int16")
    target_doy_values = np.arange(1, 366, dtype=np.int16)
    target_doys = xr.DataArray(
        target_doy_values,
        dims="doy",
        coords={"doy": target_doy_values},
    )

    half_before = (n_days - 1) // 2
    half_after = n_days // 2
    circular_delta = ((doys - target_doys + 182) % 365) - 182
    in_window = (circular_delta >= -half_before) & (
        circular_delta <= half_after
    )
    climatology_ds = _winsorized_mean_over_dim(
        ds_reference.where(in_window),
        dim="time",
        lower_quantile=RUNNING_CLIMATOLOGY_WINSOR_LOWER_QUANTILE,
        upper_quantile=RUNNING_CLIMATOLOGY_WINSOR_UPPER_QUANTILE,
    )

    doy_size = climatology_ds.sizes.get("doy", 0)
    dummy_dates = pd.date_range("2001-01-01", periods=doy_size, freq="D")
    return climatology_ds.assign_coords(
        dummy_ref_datetime=("doy", dummy_dates)
    )


def _write_climatology_products(
    climatology_ds: xr.Dataset,
    suffix: str = "",
    output_dir: Optional[Path] = None,
    output_group: Optional[str] = None,
    mean_climatology: Optional[xr.Dataset] = None,
    monthly_climatology: Optional[xr.Dataset] = None,
    seasonal_climatology: Optional[xr.Dataset] = None,
) -> None:
    output_dir = output_dir or BASE_DATA_PATH

    _write_dataset(
        climatology_ds.drop_vars("dummy_ref_datetime", errors="ignore"),
        output_dir,
        f"{LEVEL3_DATASET_STEM}_doy_climatology{suffix}.nc",
        group=output_group,
    )

    if mean_climatology is None:
        mean_climatology = climatology_ds.mean(dim="doy")
    _write_dataset(
        mean_climatology,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_mean_climatology{suffix}.nc",
        group=output_group,
    )

    if monthly_climatology is None:
        monthly_climatology = climatology_ds.groupby(
            "dummy_ref_datetime.month"
        ).mean("doy")
    _write_dataset(
        monthly_climatology.drop_vars("dummy_ref_datetime", errors="ignore"),
        output_dir,
        f"{LEVEL3_DATASET_STEM}_monthly_mean_climatology{suffix}.nc",
        group=output_group,
    )

    if seasonal_climatology is None:
        seasonal_climatology = climatology_ds.groupby(
            "dummy_ref_datetime.season"
        ).mean("doy")
    _write_dataset(
        seasonal_climatology.drop_vars("dummy_ref_datetime", errors="ignore"),
        output_dir,
        f"{LEVEL3_DATASET_STEM}_seasonal_mean_climatology{suffix}.nc",
        group=output_group,
    )


def _get_site_id_from_dataset(ds: xr.Dataset) -> Optional[str]:
    """Extract a single site ID from coords/attrs if available.

    Returns None if dataset has multiple sites or no identifiable site metadata.
    """
    if "site" in ds.coords:
        site_coord = ds.coords["site"]
        if "site" in ds.dims and site_coord.size == 1:
            return str(site_coord.values[0])
        if "site" not in ds.dims and site_coord.ndim == 0:
            return str(site_coord.values.item())

    for key in ("site_id", "site", "site_name"):
        if key in ds.attrs:
            return str(ds.attrs[key])

    return None


def _get_instrument_id_from_dataset(ds: xr.Dataset) -> Optional[str]:
    """Extract a single instrument ID from attrs if available."""
    for key in ("instrument_id", "instrument", "instrument_name"):
        if key in ds.attrs:
            return str(ds.attrs[key])
    return None


def _parse_dtree_path(dtree_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a DataTree path like /site_id/instrument_id into IDs."""
    return _parse_dtree_leaf_path(dtree_path)


def _parse_dtree_leaf_path(
    dtree_path: str,
) -> tuple[Optional[str], Optional[str]]:
    """Parse a DataTree leaf path of the form /site_id/instrument_id."""
    if not dtree_path:
        return None, None
    parts = [p for p in dtree_path.split("/") if p]
    if not parts:
        return None, None
    site_id = parts[0]
    instrument_id = parts[1] if len(parts) > 1 else None
    return site_id, instrument_id


def _output_group_for_instrument(site_id: str, instrument_id: str) -> str:
    return f"{site_id}/{instrument_id}"


def _lock_file_path(target_path: Path) -> Path:
    return target_path.with_name(f".{target_path.name}.lock")


@contextmanager
def _exclusive_file_lock(target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_file_path(target_path)
    lock_path.touch(exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cleanup_lock_files(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    for lock_path in output_dir.glob(".*.lock"):
        try:
            lock_path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Could not remove lock file '%s'", lock_path)


def _get_ids_for_leaf_dataset(ds: xr.Dataset) -> Tuple[str, Optional[str]]:
    """Resolve site and instrument IDs for a leaf dataset.

    Priority: attrs/coords, then datatree path if provided in attrs.
    """
    site_id = _get_site_id_from_dataset(ds)
    instrument_id = _get_instrument_id_from_dataset(ds)

    if site_id is None:
        dtree_path = (
            ds.attrs.get("datatree_path")
            or ds.attrs.get("path")
            or ds.encoding.get("group")
        )
        site_from_path, instrument_from_path = _parse_dtree_path(
            str(dtree_path) if dtree_path is not None else ""
        )
        site_id = site_from_path
        if instrument_id is None:
            instrument_id = instrument_from_path

    if site_id is None:
        raise ValueError(
            "Missing site identifier. Provide a scalar 'site' coordinate or set "
            "ds.attrs['site_id'] (or ds.attrs['datatree_path'])."
        )

    return site_id, instrument_id


def _open_site_leaf_dataset(
    dataset_path: Path,
    site_id: str,
    leaf_spec: SiteLeafSpec,
) -> xr.Dataset:
    with xr.open_dataset(
        dataset_path,
        group=leaf_spec.group_path.lstrip("/"),
        engine="netcdf4",
    ) as ds:
        loaded_ds = ds.load()

    return loaded_ds.assign_attrs(
        {
            "site_id": site_id,
            "instrument_id": leaf_spec.instrument_id,
            "datatree_path": leaf_spec.group_path,
        }
    )


def _build_site_processing_tasks(
    ds_tree: xr.DataTree,
    dataset_path: Path,
    output_dir: Path,
    ground_sites,
) -> list[SiteProcessingTask]:
    site_leaves: dict[str, list[SiteLeafSpec]] = {}
    all_site_coordinates = get_site_coordinate_lookup(ground_sites)

    for path, (node,) in group_subtrees(ds_tree):
        if node.children:
            continue

        site_id, instrument_id = _parse_dtree_leaf_path(path)
        if site_id is None or instrument_id is None:
            raise ValueError(
                "Missing site or instrument identifier. Ensure DataTree leaves "
                "follow /site_id/instrument_id."
            )

        if site_id not in all_site_coordinates:
            logger.warning(
                "Skipping site '%s' because it is missing from the ground-site metadata.",
                site_id,
            )
            continue

        site_leaves.setdefault(site_id, []).append(
            SiteLeafSpec(
                group_path=path,
                instrument_id=instrument_id,
            )
        )

    tasks: list[SiteProcessingTask] = []
    for site_id in sorted(site_leaves):
        leaf_specs = tuple(
            sorted(site_leaves[site_id], key=lambda spec: spec.group_path)
        )
        tasks.append(
            SiteProcessingTask(
                dataset_path=dataset_path,
                output_dir=output_dir,
                site_id=site_id,
                site_coordinate_lookup={
                    site_id: all_site_coordinates[site_id]
                },
                leaf_specs=leaf_specs,
            )
        )

    return tasks


def _process_site_task(task: SiteProcessingTask) -> Optional[str]:
    if DATASET_TO_PROCESS == "alc":
        return _process_alc_site_task(task)
    return _process_site_task_with_super_series(task)


def _process_site_task_with_super_series(
    task: SiteProcessingTask,
) -> Optional[str]:
    logger.info(
        "Processing site '%s' with %d instrument leaves",
        task.site_id,
        len(task.leaf_specs),
    )

    leaf_datasets: list[tuple[SiteLeafSpec, xr.Dataset]] = [
        (
            leaf_spec,
            _open_site_leaf_dataset(
                task.dataset_path,
                task.site_id,
                leaf_spec,
            ),
        )
        for leaf_spec in task.leaf_specs
    ]

    try:
        combined_ds = _combine_site_instruments(
            task.site_id,
            [(leaf_spec.instrument_id, ds) for leaf_spec, ds in leaf_datasets],
        )
    except ValueError as exc:
        logger.warning("Skipping site '%s': %s", task.site_id, exc)
        return None

    processed, climatology_map = _process_time_resolved_dataset(
        combined_ds,
        ground_sites=None,
        output_dir=task.output_dir,
        output_group=f"{task.site_id}/super",
        return_climatology=True,
        site_coordinate_lookup=task.site_coordinate_lookup,
        use_dask=False,
    )
    if not processed or climatology_map is None:
        return None

    for leaf_spec, ds in leaf_datasets:
        instrument_group = _output_group_for_instrument(
            task.site_id,
            leaf_spec.instrument_id,
        )
        logger.info(
            "Processing instrument series for site '%s', instrument '%s'",
            task.site_id,
            leaf_spec.instrument_id,
        )
        _process_time_resolved_dataset(
            ds,
            ground_sites=None,
            output_dir=task.output_dir,
            output_group=instrument_group,
            enforce_min_days=False,
            climatology_override=climatology_map,
            site_coordinate_lookup=task.site_coordinate_lookup,
            use_dask=False,
        )

    return task.site_id


def _build_shared_alc_climatology(
    site_id: str,
    ds_daily_by_instrument: dict[str, dict[str, xr.Dataset]],
) -> tuple[
    dict[str, dict[str, xr.Dataset]],
    dict[str, xr.Dataset],
    dict[str, xr.Dataset],
    dict[str, tuple[xr.Dataset, xr.Dataset, xr.Dataset]],
]:
    instrument_baselines: dict[str, dict[str, xr.Dataset]] = {
        instrument_id: {} for instrument_id in ds_daily_by_instrument
    }
    shared_zero_centered: dict[str, xr.Dataset] = {}
    shared_absolute: dict[str, xr.Dataset] = {}
    shared_summaries: dict[str, tuple[xr.Dataset, xr.Dataset, xr.Dataset]] = {}

    for condition, suffix in CONDITION_SUFFIX_MAP.items():
        demeaned_datasets: list[xr.Dataset] = []
        absolute_datasets: list[xr.Dataset] = []

        for (
            instrument_id,
            datasets_by_condition,
        ) in ds_daily_by_instrument.items():
            ds_daily = datasets_by_condition[condition]
            baseline = _baseline_from_reference_period(ds_daily)
            instrument_baselines[instrument_id][condition] = baseline
            demeaned_datasets.append(ds_daily - baseline)
            absolute_datasets.append(ds_daily)

        pooled_demeaned = _concatenate_time_series_datasets(
            demeaned_datasets,
            attrs={"site_id": site_id, "climatology_role": "shared_demeaned"},
        )
        shared_climatology = _compute_doy_running_climatology_dataset(
            pooled_demeaned,
            n_days=N_RUNNING_CLIMATOLOGY_DAYS,
        )
        shared_climatology = _center_climatology_on_zero(shared_climatology)
        shared_zero_centered[condition] = shared_climatology

        pooled_absolute = _concatenate_time_series_datasets(
            absolute_datasets,
            attrs={"site_id": site_id, "climatology_role": "shared_absolute"},
        )
        mean_climatology, monthly_climatology, seasonal_climatology = (
            _summary_climatologies_from_reference_period(pooled_absolute)
        )
        shared_summaries[condition] = (
            mean_climatology,
            monthly_climatology,
            seasonal_climatology,
        )
        shared_absolute[condition] = shared_climatology + mean_climatology

        shared_zero_centered[condition].attrs.update(
            {
                "site_id": site_id,
                "climatology_role": "shared_zero_centered",
                "comment": (
                    "Shared seasonal climatology estimated from pooled daily "
                    f"instrument anomalies{suffix} after removing instrument-specific "
                    "reference-period means."
                ),
            }
        )
        shared_absolute[condition].attrs.update(
            {
                "site_id": site_id,
                "climatology_role": "shared_site_climatology",
                "comment": (
                    "Site-level shared climatology for relative normalization, "
                    "formed from the pooled site baseline mean plus the shared "
                    f"zero-centered seasonal cycle{suffix}."
                ),
            }
        )

    return (
        instrument_baselines,
        shared_zero_centered,
        shared_absolute,
        shared_summaries,
    )


def _write_shared_alc_climatology_products(
    site_id: str,
    shared_absolute_climatology: dict[str, xr.Dataset],
    shared_summaries: dict[str, tuple[xr.Dataset, xr.Dataset, xr.Dataset]],
    output_dir: Path,
) -> None:
    output_group = _shared_climatology_group(site_id)
    for condition, suffix in CONDITION_SUFFIX_MAP.items():
        mean_climatology, monthly_climatology, seasonal_climatology = (
            shared_summaries[condition]
        )
        _write_climatology_products(
            shared_absolute_climatology[condition],
            suffix=suffix,
            output_dir=output_dir,
            output_group=output_group,
            mean_climatology=mean_climatology,
            monthly_climatology=monthly_climatology,
            seasonal_climatology=seasonal_climatology,
        )


def _process_alc_site_task(task: SiteProcessingTask) -> Optional[str]:
    logger.info(
        "Processing ALC site '%s' with %d instrument leaves",
        task.site_id,
        len(task.leaf_specs),
    )

    leaf_datasets: list[tuple[SiteLeafSpec, xr.Dataset]] = []
    ds_daily_by_instrument: dict[str, dict[str, xr.Dataset]] = {}

    for leaf_spec in task.leaf_specs:
        if leaf_spec.instrument_id == "unknown_ceilometer":
            logger.info(
                "Skipping site '%s' instrument '%s' in shared ALC processing.",
                task.site_id,
                leaf_spec.instrument_id,
            )
            continue

        ds_leaf = _open_site_leaf_dataset(
            task.dataset_path,
            task.site_id,
            leaf_spec,
        )
        try:
            ds_daily_by_condition = _aggregate_daily_condition_datasets(
                ds_leaf,
                ground_sites=None,
                site_coordinate_lookup=task.site_coordinate_lookup,
                use_dask=False,
            )
            valid_days = _count_valid_days(ds_daily_by_condition["all"])
            if valid_days < ALC_MIN_INSTRUMENT_VALID_DAYS:
                logger.warning(
                    "Skipping instrument '%s' for site '%s' due to insufficient daily coverage: %d < %d valid days.",
                    leaf_spec.instrument_id,
                    task.site_id,
                    valid_days,
                    ALC_MIN_INSTRUMENT_VALID_DAYS,
                )
                continue

            ds_daily_by_instrument[leaf_spec.instrument_id] = ds_daily_by_condition
        except ValueError as exc:
            logger.warning(
                "Skipping instrument '%s' for site '%s': %s",
                leaf_spec.instrument_id,
                task.site_id,
                exc,
            )
            continue
        leaf_datasets.append((leaf_spec, ds_leaf))

    if not ds_daily_by_instrument:
        logger.warning(
            "Skipping site '%s': no valid instrument datasets.", task.site_id
        )
        return None

    valid_days = _count_valid_days_across_datasets(
        [datasets["all"] for datasets in ds_daily_by_instrument.values()]
    )
    if valid_days < MIN_VALID_DAYS:
        logger.warning(
            "Skipping site '%s' due to insufficient pooled daily coverage: %d < %d valid days.",
            task.site_id,
            valid_days,
            MIN_VALID_DAYS,
        )
        return None

    try:
        (
            instrument_baselines,
            shared_zero_centered,
            shared_absolute,
            shared_summaries,
        ) = _build_shared_alc_climatology(task.site_id, ds_daily_by_instrument)
    except ValueError as exc:
        logger.warning("Skipping site '%s': %s", task.site_id, exc)
        return None

    _write_shared_alc_climatology_products(
        task.site_id,
        shared_absolute,
        shared_summaries,
        task.output_dir,
    )

    for leaf_spec, _ in leaf_datasets:
        instrument_id = leaf_spec.instrument_id
        instrument_group = _output_group_for_instrument(
            task.site_id, instrument_id
        )
        logger.info(
            "Processing instrument series for site '%s', instrument '%s'",
            task.site_id,
            instrument_id,
        )
        ds_daily_by_condition = ds_daily_by_instrument[instrument_id]
        instrument_climatology = {
            condition: shared_zero_centered[condition]
            + instrument_baselines[instrument_id][condition]
            for condition in CONDITION_SUFFIX_MAP
        }

        _write_daily_condition_products(
            ds_daily_by_condition,
            task.output_dir,
            output_group=instrument_group,
        )
        _run_daily_postprocessing(
            ds_daily_by_condition,
            instrument_climatology,
            task.output_dir,
            output_group=instrument_group,
        )

    return task.site_id


def _build_instrument_priority_map() -> dict[str, int]:
    """Build priority mapping from CEILOMETERS ordering."""
    priority: dict[str, int] = {}
    for idx, ceilometer_cls in enumerate(CEILOMETERS):
        names = {ceilometer_cls.__name__.lower(), ceilometer_cls.name.lower()}
        names.update({n.lower() for n in ceilometer_cls.alternative_names})
        for name in names:
            priority.setdefault(name, idx)
    return priority


def _normalize_instrument_id(instrument_id: Optional[str]) -> Optional[str]:
    if instrument_id is None:
        return None
    return instrument_id.lower().replace(" ", "").replace("-", "")


def _instrument_sort_key(
    instrument_id: Optional[str],
    priority_map: dict[str, int],
) -> tuple[int, str]:
    if instrument_id is None:
        return (len(priority_map) + 1, "")

    raw = instrument_id.lower()
    normalized = _normalize_instrument_id(instrument_id)
    if raw in priority_map:
        return (priority_map[raw], raw)
    if normalized and normalized in priority_map:
        return (priority_map[normalized], normalized)
    return (len(priority_map) + 1, raw)

def _combine_site_instruments(
    site_id: str,
    instrument_datasets: list[tuple[Optional[str], xr.Dataset]],
) -> xr.Dataset:
    """Combine multiple instrument datasets into a single time series.

    Uses CEILOMETERS order as priority; higher priority instruments fill first,
    and lower priority instruments fill remaining gaps.
    """
    if not instrument_datasets:
        raise ValueError(f"No datasets available for site '{site_id}'.")

    instrument_datasets = [
        ds for ds in instrument_datasets if ds[0] != "unknown_ceilometer"
    ]

    priority_map = _build_instrument_priority_map()
    sorted_items = sorted(
        instrument_datasets,
        key=lambda item: _instrument_sort_key(item[0], priority_map),
    )

    combined = sorted_items[0][1]
    for _, ds in sorted_items[1:]:
        combined = combined.combine_first(ds)

    combined = combined.copy(deep=False).assign_attrs({"site_id": site_id})
    return combined


def _write_dataset(
    ds: xr.Dataset,
    output_dir: Path,
    filename: str,
    group: Optional[str] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    netcdf_writer: Any
    with _exclusive_file_lock(output_path):
        output_exists = output_path.exists() and output_path.stat().st_size > 0
        if group:
            mode = "a" if output_exists else "w"
            ds = ds.drop_vars("dummy_ref_datetime", errors="ignore")
            netcdf_writer = ds
            netcdf_writer.to_netcdf(
                path=str(output_path),
                mode=mode,
                group=group,
                engine="netcdf4",
            )
        else:
            ds = ds.drop_vars("dummy_ref_datetime", errors="ignore")
            netcdf_writer = ds
            netcdf_writer.to_netcdf(
                path=str(output_path),
                mode="w",
                engine="netcdf4",
            )


def _chunk_time_resolved_dataset(
    ds_time_resolved: xr.Dataset,
    use_dask: bool = False,
) -> xr.Dataset:
    if not use_dask:
        return ds_time_resolved.load()

    chunks = {"time": -1}
    if "site" in ds_time_resolved.dims:
        chunks["site"] = 1
    return ds_time_resolved.chunk(chunks).load()


def _drop_ceres_qc_variables(ds_time_resolved: xr.Dataset) -> xr.Dataset:
    """Drop CERES level2 QC-only intermediate variables before level3 aggregation.

    upper_level_cloud_cover_obs/adj are only needed to compute the level2
    non-obscured cloud cover ratio and should not be carried through to
    level3 aggregated outputs.
    """
    if DATASET_TO_PROCESS != "ceres":
        return ds_time_resolved
    return ds_time_resolved.drop_vars(
        list(CERES_LEVEL3_EXCLUDED_VARIABLES), errors="ignore"
    )


def _aggregated_variable_name(var_name: str) -> str:
    if var_name in AGGREGATED_FLAG_VARIABLE_METADATA:
        return AGGREGATED_FLAG_VARIABLE_METADATA[var_name]["name"]
    if var_name.endswith("_flag"):
        return f"{var_name[:-5]}_cover"
    return var_name


def _flag_comment_to_fraction_comment(comment: str) -> str:
    replacements = (
        (
            "Boolean flag indicating presence of ",
            "Fraction of samples with ",
        ),
        (
            "Boolean flag indicating profiles that were ",
            "Fraction of samples that were ",
        ),
        ("Boolean flag indicating ", "Fraction of samples indicating "),
        ("Boolean flag ", "Fraction of samples "),
    )
    for old, new in replacements:
        if old in comment:
            return comment.replace(old, new)
    return comment.replace(" flag", " fraction").replace(" Flag", " Fraction")


def _rename_aggregated_flag_variables(ds: xr.Dataset) -> xr.Dataset:
    rename_map: dict[str, str] = {}
    for var_name in map(str, ds.data_vars):
        new_name = _aggregated_variable_name(var_name)
        if new_name != var_name:
            rename_map[var_name] = new_name

    if not rename_map:
        return ds

    renamed_ds = ds.rename(rename_map)
    for old_name, new_name in rename_map.items():
        old_var_name = str(old_name)
        new_var_name = str(new_name)
        attrs = dict(renamed_ds[new_var_name].attrs)
        metadata_override: dict[str, str] = (
            AGGREGATED_FLAG_VARIABLE_METADATA[old_var_name]
            if old_var_name in AGGREGATED_FLAG_VARIABLE_METADATA
            else {}
        )

        long_name = metadata_override.get("long_name") or attrs.get(
            "long_name"
        )
        if isinstance(long_name, str):
            attrs["long_name"] = long_name.replace(
                " flag", " fraction"
            ).replace(" Flag", " Fraction")

        comment = metadata_override.get("comment") or attrs.get("comment")
        if isinstance(comment, str):
            attrs["comment"] = _flag_comment_to_fraction_comment(comment)

        attrs["units"] = "1"
        renamed_ds[new_var_name].attrs = attrs

    return renamed_ds


def _count_valid_days(ds_daily: xr.Dataset) -> int:
    """Count days with at least one valid value for a representative variable."""
    if not ds_daily.data_vars:
        return 0

    var_name = next(iter(ds_daily.data_vars))
    da = ds_daily[var_name]
    reduce_dims = [dim for dim in da.dims if dim != "time"]
    if reduce_dims:
        valid_per_time = da.notnull().any(dim=reduce_dims)
    else:
        valid_per_time = da.notnull()

    return int(valid_per_time.sum(dim="time").item())


def _representative_time_series_var_name(ds_time_resolved: xr.Dataset) -> str | None:
    """Choose a representative variable for time-sample availability checks."""
    if not ds_time_resolved.data_vars:
        return None

    preferred_names = ("low_cloud_flag", "cloud_cover")
    for var_name in preferred_names:
        if var_name in ds_time_resolved.data_vars:
            return var_name

    for var_name in ds_time_resolved.data_vars:
        if var_name.endswith("_flag"):
            return var_name

    return next(iter(ds_time_resolved.data_vars))


def _daily_sample_counts(ds_time_resolved: xr.Dataset) -> xr.DataArray | None:
    """Count available samples per day using a representative time-series variable."""
    var_name = _representative_time_series_var_name(ds_time_resolved)
    if var_name is None:
        return None

    da = ds_time_resolved[var_name]
    reduce_dims = [dim for dim in da.dims if dim != "time"]
    if reduce_dims:
        valid_per_sample = da.notnull().any(dim=reduce_dims)
    else:
        valid_per_sample = da.notnull()

    return valid_per_sample.resample(time="D").sum()


def aggregate_time_resolved_to_daily(
    ds_time_resolved: xr.Dataset, sza_da: xr.DataArray
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Aggregate time-resolved samples to daily values for all, daytime, and nighttime.

    Args:
        ds_time_resolved: Dataset with single sample times on the 'time' dimension
        sza_da: Solar zenith angle DataArray

    Returns:
        Tuple of (ds_daily_all, ds_daily_daytime, ds_daily_nighttime)
    """
    logger.info("Aggregating single-sample data to daily values...")

    # All observations
    ds_daily_all = ds_time_resolved.resample(time="D").mean()

    # These day/night thresholds are intentionally independent of the stricter
    # autocalibration nighttime screen used for lidar-ratio candidates.
    # Daytime observations (SZA <= 85°)
    ds_daytime = ds_time_resolved.where(sza_da <= 85)
    ds_daily_daytime = ds_daytime.resample(time="D").mean()

    # Nighttime observations (SZA > 90°)
    ds_nighttime = ds_time_resolved.where(sza_da > 90)
    ds_daily_nighttime = ds_nighttime.resample(time="D").mean()

    # Mask days with insufficient valid samples for ALC datasets
    if DATASET_TO_PROCESS == "alc":
        all_daily_sample_counts = _daily_sample_counts(ds_time_resolved)
        if all_daily_sample_counts is not None:
            ds_daily_all = ds_daily_all.where(
                all_daily_sample_counts >= ALC_MIN_DAILY_SAMPLES,
                drop=True,
            )

        daytime_daily_sample_counts = _daily_sample_counts(ds_daytime)
        if daytime_daily_sample_counts is not None:
            ds_daily_daytime = ds_daily_daytime.where(
                daytime_daily_sample_counts >= ALC_MIN_DAILY_SAMPLES,
                drop=True,
            )

        nighttime_daily_sample_counts = _daily_sample_counts(ds_nighttime)
        if nighttime_daily_sample_counts is not None:
            ds_daily_nighttime = ds_daily_nighttime.where(
                nighttime_daily_sample_counts >= ALC_MIN_DAILY_SAMPLES,
                drop=True,
            )

    ds_daily_all = _rename_aggregated_flag_variables(ds_daily_all)
    ds_daily_daytime = _rename_aggregated_flag_variables(ds_daily_daytime)
    ds_daily_nighttime = _rename_aggregated_flag_variables(ds_daily_nighttime)

    logger.info("Daily aggregation complete")
    return ds_daily_all, ds_daily_daytime, ds_daily_nighttime


def write_coarser_raw_aggregates(
    ds_daily_raw: xr.Dataset,
    suffix: str = "",
    output_dir: Optional[Path] = None,
    output_group: Optional[str] = None,
) -> None:
    """Write direct multi-day and coarser means from daily raw data."""
    logger.info(
        f"Writing 2-day, 3-day, weekly, monthly, seasonal, and yearly raw aggregates{suffix}..."
    )

    output_dir = output_dir or BASE_DATA_PATH

    ds_2day_raw = ds_daily_raw.resample(time="2D").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_2day_raw,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_2day_raw{suffix}.nc",
        group=output_group,
    )

    ds_3day_raw = ds_daily_raw.resample(time="3D").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_3day_raw,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_3day_raw{suffix}.nc",
        group=output_group,
    )

    ds_weekly_raw = ds_daily_raw.resample(time=WEEKLY_RESAMPLE_FREQ).map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_weekly_raw,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_weekly_raw{suffix}.nc",
        group=output_group,
    )

    ds_monthly_raw = ds_daily_raw.resample(time="MS").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_monthly_raw,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_monthly_raw{suffix}.nc",
        group=output_group,
    )

    ds_seasonal_raw = ds_daily_raw.resample(time="QS-DEC").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_seasonal_raw,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_seasonal_raw{suffix}.nc",
        group=output_group,
    )

    ds_yearly_raw = ds_daily_raw.resample(time="YS").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_yearly_raw,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_yearly_raw{suffix}.nc",
        group=output_group,
    )


def compute_doy_running_climatology(
    ds_daily: xr.Dataset,
    n_days: int,
    suffix: str = "",
    output_dir: Optional[Path] = None,
    output_group: Optional[str] = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute day-of-year N-day running climatology and anomaly from daily data.

    This approach computes the mean cloud fraction for each day-of-year
    with N-day running average (365 values per site), then subtracts this
    climatology from the daily data to get anomalies.

    Additionally computes and saves:
    - Grand mean climatology (averaged over all doys)
    - Monthly mean climatology (averaged over doys for each month)
    - Seasonal mean climatology (averaged over doys for each season)

    Args:
        ds_daily: Daily dataset with 'time' dimension and data variables
        n_days: Days to include in the running mean
        suffix: Suffix for output filenames (e.g., "_daytime", "_nighttime")

    Returns:
        Tuple of (climatology dataset, anomaly dataset)
    """
    logger.info(
        f"Computing day-of-year {n_days}-day running climatology{suffix}..."
    )
    climatology_ds = _compute_doy_running_climatology_dataset(ds_daily, n_days)

    logger.info(f"Computing grand mean climatology{suffix}...")
    logger.info(f"Computing monthly mean climatology{suffix}...")
    logger.info(f"Computing seasonal mean climatology{suffix}...")
    _write_climatology_products(
        climatology_ds,
        suffix=suffix,
        output_dir=output_dir,
        output_group=output_group,
    )

    logger.info(f"Computing anomaly from climatology{suffix}...")

    # Compute anomaly by subtracting climatology from observations
    doys = ds_daily.time.dt_noleap.dayofyear

    anomaly_ds = xr.Dataset()
    for var_name in ds_daily.data_vars:
        # Select climatology for each time point
        clim_aligned = climatology_ds[var_name].sel(doy=doys)

        # Compute anomaly: observation - climatology
        anomaly_ds[var_name] = ds_daily[var_name] - clim_aligned

        # Update attributes
        anomaly_ds[var_name].attrs.update(
            {
                "long_name": f"{var_name} - anomaly from day-of-year {n_days}-day climatology{suffix}",
                "method": "observation - climatology",
            }
        )

    logger.info(f"Climatology computation complete{suffix}")

    return climatology_ds, anomaly_ds


def _prepare_time_resolved_dataset_for_level3(
    ds_time_resolved: xr.Dataset,
    use_dask: bool = False,
) -> xr.Dataset:
    ds_time_resolved = _chunk_time_resolved_dataset(
        ds_time_resolved,
        use_dask=use_dask,
    )
    ds_time_resolved = ds_time_resolved.sel(
        time=~(
            (ds_time_resolved.time.dt.month == 2)
            & (ds_time_resolved.time.dt.day == 29)
        )
    )
    return ds_time_resolved


def _aggregate_daily_condition_datasets(
    ds_time_resolved: xr.Dataset,
    ground_sites=None,
    site_coordinate_lookup: Optional[dict[str, tuple[float, float]]] = None,
    use_dask: bool = False,
) -> dict[str, xr.Dataset]:
    prepared_ds = _prepare_time_resolved_dataset_for_level3(
        ds_time_resolved,
        use_dask=use_dask,
    )
    sza_da = compute_solar_zenith_angles(
        prepared_ds,
        ground_sites=ground_sites,
        site_coordinate_lookup=site_coordinate_lookup,
    )
    sza_da = sza_da.persist() if use_dask else sza_da

    ds_daily_all, ds_daily_daytime, ds_daily_nighttime = (
        aggregate_time_resolved_to_daily(prepared_ds, sza_da)
    )

    return {
        "all": ds_daily_all.persist() if use_dask else ds_daily_all,
        "daytime": ds_daily_daytime.persist()
        if use_dask
        else ds_daily_daytime,
        "nighttime": ds_daily_nighttime.persist()
        if use_dask
        else ds_daily_nighttime,
    }


def _write_daily_condition_products(
    ds_daily_by_condition: dict[str, xr.Dataset],
    output_dir: Path,
    output_group: Optional[str] = None,
) -> None:
    _write_dataset(
        ds_daily_by_condition["all"],
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_raw.nc",
        group=output_group,
    )
    _write_dataset(
        ds_daily_by_condition["daytime"],
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_raw_daytime.nc",
        group=output_group,
    )
    _write_dataset(
        ds_daily_by_condition["nighttime"],
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_raw_nighttime.nc",
        group=output_group,
    )

    for condition, suffix in CONDITION_SUFFIX_MAP.items():
        write_coarser_raw_aggregates(
            ds_daily_by_condition[condition],
            suffix=suffix,
            output_dir=output_dir,
            output_group=output_group,
        )


def _run_daily_postprocessing(
    ds_daily_by_condition: dict[str, xr.Dataset],
    climatology_by_condition: dict[str, xr.Dataset],
    output_dir: Path,
    output_group: Optional[str] = None,
) -> None:
    anomalies: dict[str, xr.Dataset] = {}
    for condition, suffix in CONDITION_SUFFIX_MAP.items():
        anomalies[condition] = compute_anomaly_from_climatology(
            ds_daily_by_condition[condition],
            climatology_by_condition[condition],
            suffix=suffix,
        )
        aggregate_and_reconstruct_from_anomaly(
            anomalies[condition],
            climatology_ds=climatology_by_condition[condition],
            suffix=suffix,
            output_dir=output_dir,
            output_group=output_group,
        )
        aggregate_and_reconstruct_by_season(
            anomalies[condition],
            climatology_ds=climatology_by_condition[condition],
            suffix=suffix,
            output_dir=output_dir,
            output_group=output_group,
        )


def _count_valid_days_across_datasets(datasets: list[xr.Dataset]) -> int:
    if not datasets:
        return 0

    valid_masks: list[xr.DataArray] = []
    for ds_daily in datasets:
        if not ds_daily.data_vars:
            continue
        var_name = next(iter(ds_daily.data_vars))
        da = ds_daily[var_name]
        reduce_dims = [dim for dim in da.dims if dim != "time"]
        if reduce_dims:
            valid_mask = da.notnull().any(dim=reduce_dims)
        else:
            valid_mask = da.notnull()
        valid_masks.append(valid_mask)

    if not valid_masks:
        return 0

    combined_valid = xr.concat(valid_masks, dim="source").any(dim="source")
    return int(combined_valid.sum(dim="time").item())


def reconstruct_from_climatology_and_anomaly(
    climatology_ds: xr.Dataset,
    anomaly_ds: xr.Dataset,
    include_seasonal: bool = False,
) -> xr.Dataset:
    """Reconstruct time series from climatology + aggregated anomaly.

    This function adds aggregated anomaly back to the appropriate climatological mean,
    depending on which cycles should be preserved.

    Reconstruction strategy by temporal scale:
    - Yearly: grand mean + anomaly (no seasonal cycle)
    - Monthly/Seasonal: seasonal mean + anomaly (preserve seasonal cycle)
    - Daily: full climatology (doy) + anomaly (preserve seasonal cycle)

    Args:
        climatology_ds: Dataset containing climatology (doy × site)
        anomaly_ds: Dataset containing aggregated anomaly at any temporal resolution
        include_seasonal: If True, include day-of-year variation (for daily/monthly/seasonal aggregates)

    Returns:
        Dataset with reconstructed time series
    """
    # Start with grand mean (averaged over all doys)
    grand_mean = climatology_ds.mean(dim="doy")

    # Broadcast grand mean to match anomaly shape
    reconstructed_ds = grand_mean + 0 * anomaly_ds

    # Add seasonal component if requested (doy variation)
    if include_seasonal:
        times = anomaly_ds["time"]
        daysofyear = times.dt_noleap.dayofyear

        # Align with time series
        seasonal_component = xr.Dataset()
        for var_name in anomaly_ds.data_vars:
            seasonal_component[var_name] = (
                climatology_ds[var_name].sel(doy=daysofyear)
                - grand_mean[var_name]
            )

        reconstructed_ds = reconstructed_ds + seasonal_component

    # Add anomaly
    reconstructed_ds = reconstructed_ds + anomaly_ds

    # Update attributes
    components = ["grand_mean"]
    if include_seasonal:
        components.append("seasonal")
    components.append("anomaly")

    for var_name in reconstructed_ds.data_vars:
        reconstructed_ds[var_name].attrs.update(
            {
                "long_name": f"{var_name} - reconstructed ({' + '.join(components)})",
                "description": "Robust statistic using climatology to handle temporal gaps",
                "method": " + ".join(components),
            }
        )

    return reconstructed_ds


def aggregate_and_reconstruct_from_anomaly(
    ds_daily_anomaly: xr.Dataset,
    climatology_ds: xr.Dataset,
    suffix: str = "",
    output_dir: Optional[Path] = None,
    output_group: Optional[str] = None,
) -> None:
    """Aggregate daily anomaly to multiple temporal resolutions and reconstruct time series.

    This function handles the complete workflow of:
    1. Saving daily anomaly
    2. Aggregating anomaly to monthly, seasonal, yearly, and running yearly
    3. Reconstructing time series with appropriate seasonal components for each resolution

    Reconstruction strategy by temporal scale:
    - Daily: baseline + seasonal + anomaly (full model)
    - Monthly: baseline + seasonal + anomaly (preserve seasonal cycle)
    - Seasonal: baseline + seasonal + anomaly (preserve seasonal cycle)
    - Yearly: baseline + anomaly (removes seasonal cycle, shows long-term trends)

    Args:
        ds_daily_anomaly: Dataset containing daily anomaly
        climatology_ds: Dataset containing day-of-year climatology
        suffix: Suffix for output filenames (e.g., "_daytime", "_nighttime")
    """
    logger.info(f"Aggregating daily anomaly and reconstructing{suffix}...")

    # Save daily anomaly
    output_dir = output_dir or BASE_DATA_PATH
    _write_dataset(
        ds_daily_anomaly,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_anomaly{suffix}.nc",
        group=output_group,
    )

    # Aggregate anomaly to standard temporal resolutions
    logger.info(
        f"  Aggregating anomaly to 2-day, 3-day, weekly, monthly, seasonal, and yearly{suffix}..."
    )

    ds_2day_anomaly = ds_daily_anomaly.resample(time="2D").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_2day_anomaly,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_2day_anomaly{suffix}.nc",
        group=output_group,
    )

    ds_3day_anomaly = ds_daily_anomaly.resample(time="3D").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_3day_anomaly,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_3day_anomaly{suffix}.nc",
        group=output_group,
    )

    ds_weekly_anomaly = ds_daily_anomaly.resample(
        time=WEEKLY_RESAMPLE_FREQ
    ).map(partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC))
    _write_dataset(
        ds_weekly_anomaly,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_weekly_anomaly{suffix}.nc",
        group=output_group,
    )

    ds_monthly_anomaly = ds_daily_anomaly.resample(time="MS").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_monthly_anomaly,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_monthly_anomaly{suffix}.nc",
        group=output_group,
    )

    ds_seasonal_anomaly = ds_daily_anomaly.resample(time="QS-DEC").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_seasonal_anomaly,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_seasonal_anomaly{suffix}.nc",
        group=output_group,
    )

    ds_yearly_anomaly = ds_daily_anomaly.resample(time="YS").map(
        partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
    )
    _write_dataset(
        ds_yearly_anomaly,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_yearly_anomaly{suffix}.nc",
        group=output_group,
    )

    window_size = min(365, ds_daily_anomaly.sizes.get("time", 0))
    if window_size >= 2:
        min_periods = max(1, int(window_size * MIN_DATA_FRAC_RUNNING))
        ds_yearly_running_anomaly_daily_freq = ds_daily_anomaly.rolling(
            time=window_size, center=True, min_periods=min_periods
        ).mean()
        _write_dataset(
            ds_yearly_running_anomaly_daily_freq,
            output_dir,
            f"{LEVEL3_DATASET_STEM}_yearly_running_anomaly_daily_freq{suffix}.nc",
            group=output_group,
        )

        ds_yearly_running_anomaly_monthly_freq = (
            ds_yearly_running_anomaly_daily_freq.resample(time="MS").map(
                partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC)
            )
        )
        _write_dataset(
            ds_yearly_running_anomaly_monthly_freq,
            output_dir,
            f"{LEVEL3_DATASET_STEM}_yearly_running_anomaly_monthly_freq{suffix}.nc",
            group=output_group,
        )
    else:
        logger.warning(
            f"Skipping yearly running anomaly products{suffix} due to insufficient time length ({window_size} samples)."
        )

    # Reconstruct time series at all temporal resolutions
    logger.info(f"  Reconstructing time series{suffix}...")
    logger.info("    - Daily: baseline + seasonal + anomaly")
    logger.info("    - 2-day: baseline + seasonal + anomaly")
    logger.info("    - 3-day: baseline + seasonal + anomaly")
    logger.info("    - Weekly: baseline + seasonal + anomaly")
    logger.info("    - Monthly: baseline + seasonal + anomaly")
    logger.info("    - Seasonal: baseline + seasonal + anomaly")
    logger.info("    - Yearly: baseline + anomaly (removes seasonal cycle)")

    # Reconstruct at daily resolution
    ds_daily_reconstructed = reconstruct_from_climatology_and_anomaly(
        climatology_ds,
        ds_daily_anomaly,
        include_seasonal=True,
    )
    _write_dataset(
        ds_daily_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at daily deseasonalized resolution
    ds_daily_deseasonalized_reconstructed = (
        reconstruct_from_climatology_and_anomaly(
            climatology_ds,
            ds_daily_anomaly,
            include_seasonal=False,
        )
    )
    _write_dataset(
        ds_daily_deseasonalized_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_deseasonalized{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at 2-day resolution
    ds_2day_reconstructed = reconstruct_from_climatology_and_anomaly(
        climatology_ds,
        ds_2day_anomaly,
        include_seasonal=True,
    )
    _write_dataset(
        ds_2day_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_2day{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at 2-day deseasonalized resolution
    ds_2day_deseasonalized_reconstructed = (
        reconstruct_from_climatology_and_anomaly(
            climatology_ds,
            ds_2day_anomaly,
            include_seasonal=False,
        )
    )
    _write_dataset(
        ds_2day_deseasonalized_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_2day_deseasonalized{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at 3-day resolution
    ds_3day_reconstructed = reconstruct_from_climatology_and_anomaly(
        climatology_ds,
        ds_3day_anomaly,
        include_seasonal=True,
    )
    _write_dataset(
        ds_3day_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_3day{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at 3-day deseasonalized resolution
    ds_3day_deseasonalized_reconstructed = (
        reconstruct_from_climatology_and_anomaly(
            climatology_ds,
            ds_3day_anomaly,
            include_seasonal=False,
        )
    )
    _write_dataset(
        ds_3day_deseasonalized_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_3day_deseasonalized{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at weekly resolution
    ds_weekly_reconstructed = reconstruct_from_climatology_and_anomaly(
        climatology_ds,
        ds_weekly_anomaly,
        include_seasonal=True,
    )
    _write_dataset(
        ds_weekly_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_weekly{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at monthly resolution
    ds_monthly_reconstructed = reconstruct_from_climatology_and_anomaly(
        climatology_ds,
        ds_monthly_anomaly,
        include_seasonal=True,
    )
    _write_dataset(
        ds_monthly_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_monthly{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at monthly deseasonalized resolution
    ds_monthly_deseasonalized_reconstructed = (
        reconstruct_from_climatology_and_anomaly(
            climatology_ds,
            ds_monthly_anomaly,
            include_seasonal=False,
        )
    )
    _write_dataset(
        ds_monthly_deseasonalized_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_monthly_deseasonalized{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at seasonal resolution
    ds_seasonal_reconstructed = reconstruct_from_climatology_and_anomaly(
        climatology_ds,
        ds_seasonal_anomaly,
        include_seasonal=True,
    )
    _write_dataset(
        ds_seasonal_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_seasonal{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at seasonal deseasonalized resolution
    ds_seasonal_deseasonalized_reconstructed = (
        reconstruct_from_climatology_and_anomaly(
            climatology_ds,
            ds_seasonal_anomaly,
            include_seasonal=False,
        )
    )
    _write_dataset(
        ds_seasonal_deseasonalized_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_seasonal_deseasonalized{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at yearly resolution
    ds_yearly_reconstructed = reconstruct_from_climatology_and_anomaly(
        climatology_ds,
        ds_yearly_anomaly,
        include_seasonal=False,
    )
    _write_dataset(
        ds_yearly_reconstructed,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_yearly{suffix}.nc",
        group=output_group,
    )

    # Reconstruct at yearly running at daily resolution
    if window_size >= 2:
        ds_yearly_running_anomaly_daily_freq_reconstructed = (
            reconstruct_from_climatology_and_anomaly(
                climatology_ds,
                ds_yearly_running_anomaly_daily_freq,
                include_seasonal=False,
            )
        )
        _write_dataset(
            ds_yearly_running_anomaly_daily_freq_reconstructed,
            output_dir,
            f"{LEVEL3_DATASET_STEM}_yearly_running_daily_freq{suffix}.nc",
            group=output_group,
        )

        # Reconstruct at yearly running at monthly resolution
        ds_yearly_running_anomaly_monthly_freq_reconstructed = (
            reconstruct_from_climatology_and_anomaly(
                climatology_ds,
                ds_yearly_running_anomaly_monthly_freq,
                include_seasonal=False,
            )
        )
        _write_dataset(
            ds_yearly_running_anomaly_monthly_freq_reconstructed,
            output_dir,
            f"{LEVEL3_DATASET_STEM}_yearly_running_monthly_freq{suffix}.nc",
            group=output_group,
        )

    logger.info(f"  Completed aggregation and reconstruction{suffix}")


def aggregate_and_reconstruct_by_season(
    ds_daily_anomaly: xr.Dataset,
    climatology_ds: xr.Dataset,
    suffix: str = "",
    output_dir: Optional[Path] = None,
    output_group: Optional[str] = None,
) -> None:
    """Aggregate daily anomaly by season for yearly periods and reconstruct.

    This function aggregates anomaly grouped by year and season, then reconstructs
    by adding back the appropriate seasonal components.

    Args:
        ds_daily_anomaly: Dataset containing daily anomaly
        climatology_ds: Dataset containing day-of-year climatology
        suffix: Suffix for output filenames (e.g., "_daytime", "_nighttime")
    """
    logger.info(
        f"Aggregating anomaly by season (yearly time series) and reconstructing{suffix}..."
    )

    # Compute seasonal mean climatology
    clim_seasonal = climatology_ds.groupby("dummy_ref_datetime.season").mean(
        "doy"
    )

    # Aggregate anomaly by year and season
    ds_yearly_by_season_anomaly = ds_daily_anomaly.groupby(
        ["time.year", "time.season"]
    ).map(partial(mean_min_frac_along_dim, p=MIN_DATA_FRAC))

    # Reconstruct by adding seasonal climatology to anomaly
    ds_yearly_by_season_reconstructed = (
        clim_seasonal + ds_yearly_by_season_anomaly
    )

    # Update attributes
    for var_name in ds_yearly_by_season_reconstructed.data_vars:
        ds_yearly_by_season_reconstructed[var_name].attrs.update(
            {
                "long_name": f"{var_name} - yearly mean by season{suffix}",
                "description": f"Yearly time series by season{suffix} (seasonal climatology + anomaly)",
                "method": "seasonal_climatology + anomaly",
            }
        )

    output_dir = output_dir or BASE_DATA_PATH
    _write_dataset(
        ds_yearly_by_season_reconstructed.drop_vars(
            "dummy_ref_datetime", errors="ignore"
        ),
        output_dir,
        f"{LEVEL3_DATASET_STEM}_yearly_by_season{suffix}.nc",
        group=output_group,
    )

    logger.info(
        f"  Completed yearly by season aggregation and reconstruction{suffix}"
    )


def compute_solar_zenith_angles(
    ds_time_resolved: xr.Dataset,
    ground_sites=None,
    site_coordinate_lookup: Optional[dict[str, tuple[float, float]]] = None,
) -> xr.DataArray:
    """Compute solar zenith angles for each site and time in the dataset.

    Args:
        ds_time_resolved: Dataset with single sample times on the 'time' dimension
        ground_sites: GeoDataFrame with site locations (lat/lon)

    Returns:
        DataArray of solar zenith angles with dimensions (time, site)
    """
    logger.info("Computing solar zenith angles...")
    times = ds_time_resolved["time"].data

    if "site" in ds_time_resolved.dims:
        if site_coordinate_lookup is None:
            if ground_sites is None:
                raise ValueError(
                    "ground_sites is required when site_coordinate_lookup is not provided."
                )
            site_coordinate_lookup = get_site_coordinate_lookup(
                ground_sites,
                site_ids=[
                    str(site_id) for site_id in ds_time_resolved["site"].values
                ],
            )
        return compute_solar_zenith_angle_for_sites(
            times,
            site_coordinate_lookup,
        ).assign_coords(time=ds_time_resolved.time, site=ds_time_resolved.site)

    site_id = _get_site_id_from_dataset(ds_time_resolved)
    if site_id is None:
        raise ValueError(
            "Cannot compute SZA without a site identifier. Provide a scalar 'site' "
            "coordinate or ds.attrs['site_id']."
        )

    if site_coordinate_lookup is None:
        if ground_sites is None:
            raise ValueError(
                "ground_sites is required when site_coordinate_lookup is not provided."
            )
        site_coordinate_lookup = get_site_coordinate_lookup(
            ground_sites, site_ids=[site_id]
        )

    if site_id not in site_coordinate_lookup:
        raise KeyError(f"Missing ground-site coordinates for: {site_id}")

    latitude, longitude = site_coordinate_lookup[site_id]
    sza_da = compute_solar_zenith_angle_for_site(
        times,
        latitude=latitude,
        longitude=longitude,
    )
    sza_da = sza_da.assign_coords(time=ds_time_resolved.time, site=site_id)
    return sza_da


def _process_time_resolved_dataset_all_only(
    ds_time_resolved: xr.Dataset,
    output_dir: Path,
    output_group: Optional[str] = None,
    enforce_min_days: bool = True,
    climatology_override: Optional[dict[str, xr.Dataset]] = None,
    return_climatology: bool = False,
    use_dask: bool = False,
) -> tuple[bool, Optional[dict[str, xr.Dataset]]]:
    """Process one input time series through the level-3 pipeline, "all" condition only.

    Used for datasets that do not support a daytime/nighttime SZA-based split
    (see SUPPORTS_DAY_NIGHT_SPLIT), such as CERES SYN1deg-Day daily composites,
    which already represent a single full-day value per site and have no
    meaningful sub-daily solar zenith angle to threshold on.
    """
    site_id = _get_site_id_from_dataset(ds_time_resolved)
    logger.info(
        f"Processing time-resolved dataset for site '{site_id}' (all only)"
        if site_id
        else "Processing time-resolved dataset (all only)"
    )
    ds_time_resolved = _chunk_time_resolved_dataset(
        ds_time_resolved,
        use_dask=use_dask,
    )

    # Step -1: Remove leap days
    ds_time_resolved = ds_time_resolved.sel(
        time=~(
            (ds_time_resolved.time.dt.month == 2)
            & (ds_time_resolved.time.dt.day == 29)
        )
    )

    # Step 2: Aggregate raw samples to daily (all only; no SZA-based split)
    ds_daily_all = ds_time_resolved.resample(time="D").mean()
    ds_daily_all = _rename_aggregated_flag_variables(ds_daily_all)
    ds_daily_all = ds_daily_all.persist() if use_dask else ds_daily_all

    valid_days = _count_valid_days(ds_daily_all)
    if enforce_min_days and valid_days < MIN_VALID_DAYS:
        logger.warning(
            "Skipping site due to insufficient daily coverage: "
            f"{valid_days} < {MIN_VALID_DAYS} valid days."
        )
        return False, None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save daily aggregated data and direct coarser raw means
    _write_dataset(
        ds_daily_all,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_raw.nc",
        group=output_group,
    )
    write_coarser_raw_aggregates(
        ds_daily_all,
        suffix="",
        output_dir=output_dir,
        output_group=output_group,
    )

    # Step 3: Compute climatology and anomaly
    if climatology_override is None:
        climatology_ds_all, anomaly_ds_all = compute_doy_running_climatology(
            ds_daily_all,
            n_days=N_RUNNING_CLIMATOLOGY_DAYS,
            suffix="",
            output_dir=output_dir,
            output_group=output_group,
        )
    else:
        climatology_ds_all = climatology_override["all"]
        anomaly_ds_all = compute_anomaly_from_climatology(
            ds_daily_all, climatology_ds_all, suffix=""
        )

    # Step 4: Aggregate anomaly and reconstruct time series
    aggregate_and_reconstruct_from_anomaly(
        anomaly_ds_all,
        climatology_ds=climatology_ds_all,
        suffix="",
        output_dir=output_dir,
        output_group=output_group,
    )

    # Step 5: Additional aggregation by season
    aggregate_and_reconstruct_by_season(
        anomaly_ds_all,
        climatology_ds=climatology_ds_all,
        suffix="",
        output_dir=output_dir,
        output_group=output_group,
    )

    climatology_result = None
    if return_climatology and climatology_override is None:
        climatology_result = {"all": climatology_ds_all}

    return True, climatology_result


def _process_time_resolved_dataset(
    ds_time_resolved: xr.Dataset,
    ground_sites,
    output_dir: Path,
    output_group: Optional[str] = None,
    enforce_min_days: bool = True,
    climatology_override: Optional[dict[str, xr.Dataset]] = None,
    return_climatology: bool = False,
    site_coordinate_lookup: Optional[dict[str, tuple[float, float]]] = None,
    use_dask: bool = False,
) -> tuple[bool, Optional[dict[str, xr.Dataset]]]:
    """Process one input time series through the level-3 aggregation pipeline."""
    if not SUPPORTS_DAY_NIGHT_SPLIT:
        return _process_time_resolved_dataset_all_only(
            ds_time_resolved,
            output_dir,
            output_group=output_group,
            enforce_min_days=enforce_min_days,
            climatology_override=climatology_override,
            return_climatology=return_climatology,
            use_dask=use_dask,
        )

    site_id = _get_site_id_from_dataset(ds_time_resolved)
    logger.info(
        f"Processing time-resolved dataset for site '{site_id}'"
        if site_id
        else "Processing time-resolved dataset"
    )
    ds_time_resolved = _chunk_time_resolved_dataset(
        ds_time_resolved,
        use_dask=use_dask,
    )

    # Step -1: Remove leap days
    ds_time_resolved = ds_time_resolved.sel(
        time=~(
            (ds_time_resolved.time.dt.month == 2)
            & (ds_time_resolved.time.dt.day == 29)
        )
    )

    # Step 1: Compute solar zenith angles
    sza_da = compute_solar_zenith_angles(
        ds_time_resolved,
        ground_sites=ground_sites,
        site_coordinate_lookup=site_coordinate_lookup,
    )

    sza_da = sza_da.persist() if use_dask else sza_da

    # Step 2: Aggregate raw samples to daily (all, daytime, nighttime)
    ds_daily_all, ds_daily_daytime, ds_daily_nighttime = (
        aggregate_time_resolved_to_daily(ds_time_resolved, sza_da)
    )
    ds_daily_all = ds_daily_all.persist() if use_dask else ds_daily_all
    ds_daily_daytime = (
        ds_daily_daytime.persist() if use_dask else ds_daily_daytime
    )
    ds_daily_nighttime = (
        ds_daily_nighttime.persist() if use_dask else ds_daily_nighttime
    )

    valid_days = _count_valid_days(ds_daily_all)
    if enforce_min_days and valid_days < MIN_VALID_DAYS:
        logger.warning(
            "Skipping site due to insufficient daily coverage: "
            f"{valid_days} < {MIN_VALID_DAYS} valid days."
        )
        return False, None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save daily aggregated data and direct coarser raw means
    _write_dataset(
        ds_daily_all,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_raw.nc",
        group=output_group,
    )
    _write_dataset(
        ds_daily_daytime,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_raw_daytime.nc",
        group=output_group,
    )
    _write_dataset(
        ds_daily_nighttime,
        output_dir,
        f"{LEVEL3_DATASET_STEM}_daily_raw_nighttime.nc",
        group=output_group,
    )
    write_coarser_raw_aggregates(
        ds_daily_all,
        suffix="",
        output_dir=output_dir,
        output_group=output_group,
    )
    write_coarser_raw_aggregates(
        ds_daily_daytime,
        suffix="_daytime",
        output_dir=output_dir,
        output_group=output_group,
    )
    write_coarser_raw_aggregates(
        ds_daily_nighttime,
        suffix="_nighttime",
        output_dir=output_dir,
        output_group=output_group,
    )

    # Step 3: Compute climatology and anomaly for each condition
    if climatology_override is None:
        climatology_ds_all, anomaly_ds_all = compute_doy_running_climatology(
            ds_daily_all,
            n_days=N_RUNNING_CLIMATOLOGY_DAYS,
            suffix="",
            output_dir=output_dir,
            output_group=output_group,
        )

        climatology_ds_daytime, anomaly_ds_daytime = (
            compute_doy_running_climatology(
                ds_daily_daytime,
                n_days=N_RUNNING_CLIMATOLOGY_DAYS,
                suffix="_daytime",
                output_dir=output_dir,
                output_group=output_group,
            )
        )

        climatology_ds_nighttime, anomaly_ds_nighttime = (
            compute_doy_running_climatology(
                ds_daily_nighttime,
                n_days=N_RUNNING_CLIMATOLOGY_DAYS,
                suffix="_nighttime",
                output_dir=output_dir,
                output_group=output_group,
            )
        )
    else:
        climatology_ds_all = climatology_override["all"]
        climatology_ds_daytime = climatology_override["daytime"]
        climatology_ds_nighttime = climatology_override["nighttime"]

        anomaly_ds_all = compute_anomaly_from_climatology(
            ds_daily_all, climatology_ds_all, suffix=""
        )
        anomaly_ds_daytime = compute_anomaly_from_climatology(
            ds_daily_daytime, climatology_ds_daytime, suffix="_daytime"
        )
        anomaly_ds_nighttime = compute_anomaly_from_climatology(
            ds_daily_nighttime, climatology_ds_nighttime, suffix="_nighttime"
        )

    # Step 4: Aggregate anomaly and reconstruct time series for each condition
    aggregate_and_reconstruct_from_anomaly(
        anomaly_ds_all,
        climatology_ds=climatology_ds_all,
        suffix="",
        output_dir=output_dir,
        output_group=output_group,
    )

    aggregate_and_reconstruct_from_anomaly(
        anomaly_ds_daytime,
        climatology_ds=climatology_ds_daytime,
        suffix="_daytime",
        output_dir=output_dir,
        output_group=output_group,
    )

    aggregate_and_reconstruct_from_anomaly(
        anomaly_ds_nighttime,
        climatology_ds=climatology_ds_nighttime,
        suffix="_nighttime",
        output_dir=output_dir,
        output_group=output_group,
    )

    # Step 5: Additional aggregations by season
    aggregate_and_reconstruct_by_season(
        anomaly_ds_all,
        climatology_ds=climatology_ds_all,
        suffix="",
        output_dir=output_dir,
        output_group=output_group,
    )

    aggregate_and_reconstruct_by_season(
        anomaly_ds_daytime,
        climatology_ds=climatology_ds_daytime,
        suffix="_daytime",
        output_dir=output_dir,
        output_group=output_group,
    )

    aggregate_and_reconstruct_by_season(
        anomaly_ds_nighttime,
        climatology_ds=climatology_ds_nighttime,
        suffix="_nighttime",
        output_dir=output_dir,
        output_group=output_group,
    )

    climatology_result = None
    if return_climatology and climatology_override is None:
        climatology_result = {
            "all": climatology_ds_all,
            "daytime": climatology_ds_daytime,
            "nighttime": climatology_ds_nighttime,
        }

    return True, climatology_result


def run() -> None:
    """Main processing pipeline for cloud cover temporal aggregation.

    This function orchestrates the complete processing workflow:
    1. Load level-2 input data and compute solar zenith angles
    2. Aggregate raw single-sample data to daily (all, daytime, nighttime) for ALC and CERES,
       or aggregate ERA5 hourly input to daily
    4. Compute day-of-year running climatology for each condition
    5. Compute anomalies from daily data
    6. Aggregate anomalies and reconstruct time series at various temporal resolutions
    """
    # Setup
    logger.info("Reading site metadata...")
    ground_sites = get_ground_sites_gdf()

    logger.info("Loading cloud cover dataset")
    dataset_path = INPUT_DATA_PATH / _level2_input_filename(
        LEVEL2_DATASET_STEM
    )

    ds_tree = xr.open_datatree(dataset_path)

    if len(ds_tree.children) > 0:
        logger.info(
            "Detected DataTree input; processing leaf datasets by site"
        )
        tasks = _build_site_processing_tasks(
            ds_tree,
            dataset_path=dataset_path,
            output_dir=BASE_DATA_PATH,
            ground_sites=ground_sites,
        )

        logger.info("Submitting %d site tasks", len(tasks))
        parallel_map(
            _process_site_task,
            tasks,
            n_jobs=N_JOBS,
            timeout=SITE_TASK_TIMEOUT,
            batch_size=max(1, min(len(tasks), N_JOBS * 2)),
        )
    else:
        logger.info(f"Launching Dask LocalCluster with {N_JOBS} workers")
        #cluster = LocalCluster(
        #    n_workers=N_JOBS, threads_per_worker=1, memory_limit="2GiB"
        #)
        #client = Client(cluster)
        #logger.info(f"Dask dashboard available at: {client.dashboard_link}")

        ds_time_resolved = ds_tree.to_dataset()
        ds_time_resolved = ds_time_resolved.sel(site=ground_sites.index)
        ds_time_resolved = _drop_ceres_qc_variables(ds_time_resolved)
        output_dir = BASE_DATA_PATH
        _process_time_resolved_dataset(
            ds_time_resolved,
            ground_sites,
            output_dir,
            use_dask=False,
        )
        #client.close()
        #cluster.close()

    _cleanup_lock_files(BASE_DATA_PATH)

    logger.info("All tasks completed.")


if __name__ == "__main__":
    setup_logging(logging.INFO)

    run()
