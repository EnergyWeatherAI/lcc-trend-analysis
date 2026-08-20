from dataclasses import dataclass
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.treenode import group_subtrees
from dotenv import load_dotenv

from lcc_trend_analysis.algorithms.trend_estimation import (
    MannKendallTrendEstimator,
)
from lcc_trend_analysis.logging import (
    get_logger,
    setup_logging,
)
from lcc_trend_analysis.parallel_processing import parallel_map
from lcc_trend_analysis.paths import get_data_paths
from lcc_trend_analysis.observations.ceres import (
    CERES_SYN1DEG_DAY_PRODUCT,
)
from lcc_trend_analysis.observations.utils import (
    get_ground_sites_gdf,
)
from lcc_trend_analysis.type_aliases import (
    GeoDataFrame,
    DataArray,
    Dataset,
)

logger: logging.Logger = get_logger(__name__)


load_dotenv()

@dataclass(frozen=True)
class DatasetVariable:
    dataset_name: str
    variable_name: str
    product: str | None = None

DATA_PATHS = get_data_paths()
N_JOBS: int = int(os.environ.get("N_JOBS", multiprocessing.cpu_count()))
CERES_PRODUCT = "SYN1deg-1Hour"

# Start cutoff date corresponding to CERES data record beginning in March 2000
RECORD_CUTOFF_DATE_START = np.datetime64("2000-03-01")
RECORD_CUTOFF_DATE_END = np.datetime64("2025-12-31")

### SETUP PARAMETERS ###
# Response data sets, i.e. cloud cover data
DATASETS: dict[str, DatasetVariable] = {
    "alc": DatasetVariable("alc", "low_cloud_cover"),
    "era5": DatasetVariable("era5", "low_cloud_cover"),
    "era5_isccp_non_obscured": DatasetVariable(
        "era5", "low_cloud_cover_isccp_non_obscured"
    ),
    "ceres_adj": DatasetVariable(
        "ceres", "low_cloud_cover_non_obscured_adj", product=CERES_PRODUCT
    ),
}

REFERENCE_DATASET: str = "alc"  # Options: keys of DATASETS

SUFFIXES: list[str] = [
    "",
    "_daytime",
    "_nighttime",
]  # Options: "", "_daytime", "_nighttime"

# Mapping of sites to granularities for which to compute trends.
GRANULARITIES: dict[str, str] = {
    "cabauw": "3day",
    "chilbolton": "3day",
    "flesland": "3day",
    "graciosa": "3day",
    "granada": "monthly",
    "kenttarova": "3day",
    "juelich": "3day",
    "kumpula": "3day",
    "lauder": "3day",
    "lindenberg": "3day",
    "nsa": "3day",
    "oslo": "3day",
    "palaiseau": "3day",
    "payerne": "3day",
    "sgp": "monthly",
    "vehmasmaki": "3day",
}
SUPPORTED_GRANULARITIES: frozenset[str] = frozenset({"daily", "3day", "weekly", "monthly"})

SIGMA_LEVELS: list[float] = [0.6827, 0.9545, 0.9973]
SIGMA_LEVEL_NAMES: list[str] = ["1σ", "2σ", "3σ"]

# Number of evenly spaced confidence levels used to trace out the full-year
# trend-slope quantile function for each dataset-site pair (see
# `TrendSlopeQuantileRecord`).
TREND_SLOPE_QUANTILE_ALPHA_POINTS: int = 20001

# Which segmentations (keys of `SEGMENTATIONS`) also get a per-segment
# trend-slope quantile function built (see `SeasonalTrendSlopeQuantileRecord`).
SEASONAL_TREND_SLOPE_QUANTILE_SEGMENTATIONS: set[str] = {"by_season"}

SAME_INSTRUMENT_ONLY_FOR_ALC: bool = True  # Whether to only compare ALC series from the same instrument when estimating trends

MIN_YEARS: int = (
    5  # Minimum number of years of valid (ALC) data for trend estimation
)

EXCLUDE_OVERLAP: bool = True  # Whether to exclude overlapping support windows for ALC samples
# Treat ALC samples within the configured window as overlapping support windows.
ALC_OVERLAP_WINDOW_DAYS: int = 2
ALC_OVERLAP_WINDOW_DAYS_BY_GRANULARITY: dict[str, int] = {
    "daily": 0,
    "3day": 2,
    "weekly": 3,
    "monthly": 13,
}
###


@dataclass(frozen=True)
class SuffixTask:
    suffix: str
    site_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrendSlopeQuantileRecord:
    """Quantile function of the full-year trend slope point estimate.

    See ``MannKendallTrendEstimator.get_trend_slope_quantile_function`` for
    how ``probability_levels``/``quantile_values`` trace out the actual
    sampling distribution of the trend estimate, as derived by inverting the
    Mann-Kendall test (not the empirical distribution of the pairwise
    slopes). This is the nonparametric distribution suitable for downstream
    Monte Carlo or Bayesian hierarchical regression.
    """

    dataset_name: str
    site: str
    probability_levels: np.ndarray
    quantile_values: np.ndarray
    n_slopes: int
    kendall_variance: float
    slope: float
    series_key: str
    n_samples: int


@dataclass(frozen=True)
class SeasonalTrendSlopeQuantileRecord:
    """Quantile function of a per-season trend slope point estimate.

    Same construction as ``TrendSlopeQuantileRecord``, but for one
    segment of a ``by_season`` segmentation (e.g. DJF/MAM/JJA/SON) rather
    than the full-year trend.
    """

    dataset_name: str
    site: str
    season: str
    probability_levels: np.ndarray
    quantile_values: np.ndarray
    n_slopes: int
    kendall_variance: float
    slope: float
    series_key: str
    n_samples: int

@dataclass
class Segmentation:
    n_segments: int
    dim_name: str
    dim: list[int] | list[str]
    climatology: str
    segmentation_func: Callable[
        [DataArray, np.ndarray], dict[int | str, np.ndarray]
    ]


SEGMENTATIONS: dict[str, Segmentation] = {
    "by_month": Segmentation(
        n_segments=12,
        dim_name="month",
        dim=list(range(1, 13)),
        climatology="monthly",
        segmentation_func=lambda reference_time_array, array_to_be_segmented: {
            month: array_to_be_segmented[
                reference_time_array.dt.month.values == month
            ]
            for month in range(1, 13)
        },
    ),
    "by_season": Segmentation(
        n_segments=4,
        dim_name="season",
        dim=["DJF", "MAM", "JJA", "SON"],
        climatology="seasonal",
        segmentation_func=lambda reference_time_array, array_to_be_segmented: {
            season: array_to_be_segmented[
                reference_time_array.dt.season.values == season
            ]
            for season in ["DJF", "MAM", "JJA", "SON"]
        },
    ),
}


def _level3_clouds_path(dataset_name: str, product: str | None = None) -> Path:
    if dataset_name == "ceres":
        return DATA_PATHS.ceres_level3_clouds(
            product or CERES_PRODUCT
        )
    return DATA_PATHS.level3_clouds(dataset_name)


def _level3_driver_path(dataset_name: str) -> Path:
    return DATA_PATHS.level3_dataset(dataset_name.split("_")[0])


def initialize_output_dataset(sites: GeoDataFrame) -> xr.Dataset:
    """Initialize the output xarray Dataset for trend estimation results."""

    rel_abs_suffixes = ["", "_relative"]

    trend_statistic_suffixes = ["", "_pvalue", "_std"]
    data_var_names_full_year = [
        f"{dataset}_trend{stat_suffix}{rel_suffix}"
        for dataset in DATASETS.keys()
        for stat_suffix in trend_statistic_suffixes
        for rel_suffix in rel_abs_suffixes
    ]

    data_vars_full_year = {
        name: (
            ("site",),
            np.full((len(sites.index),), fill_value=np.nan, dtype=np.float32),
        )
        for name in data_var_names_full_year
    }

    data_var_names_full_year_counts = [
        f"{dataset}_trend_n_samples" for dataset in DATASETS.keys()
    ]

    data_vars_full_year_counts = {
        name: (
            ("site",),
            np.full((len(sites.index),), fill_value=np.nan, dtype=np.float32),
        )
        for name in data_var_names_full_year_counts
    }

    data_var_names_full_year_bounded = [
        f"{dataset}_trend_ci{rel_suffix}"
        for dataset in DATASETS.keys()
        for rel_suffix in rel_abs_suffixes
    ]

    data_vars_full_year_bounded = {
        name: (
            ("site", "sigma_level", "bound"),
            np.full(
                (len(sites.index), len(SIGMA_LEVELS), 2),
                fill_value=np.nan,
                dtype=np.float32,
            ),
        )
        for name in data_var_names_full_year_bounded
    }

    # Segmented trend variables
    data_vars_segment = {}
    data_vars_segment_bounded = {}
    data_vars_segment_counts = {}

    for seg_name, seg_info in SEGMENTATIONS.items():
        data_var_names_segment = [
            f"{dataset}_trend_{seg_name}{stat_suffix}{rel_suffix}"
            for dataset in DATASETS.keys()
            for stat_suffix in trend_statistic_suffixes
            for rel_suffix in rel_abs_suffixes
        ]

        data_vars_segment.update(
            {
                name: (
                    ("site", seg_info.dim_name),
                    np.full(
                        (len(sites.index), seg_info.n_segments),
                        fill_value=np.nan,
                        dtype=np.float32,
                    ),  # type: ignore
                )
                for name in data_var_names_segment
            }
        )

        data_var_names_segment_counts = [
            f"{dataset}_trend_{seg_name}_n_samples" for dataset in DATASETS.keys()
        ]

        data_vars_segment_counts.update(
            {
                name: (
                    ("site", seg_info.dim_name),
                    np.full(
                        (len(sites.index), seg_info.n_segments),
                        fill_value=np.nan,
                        dtype=np.float32,
                    ),  # type: ignore
                )
                for name in data_var_names_segment_counts
            }
        )

        data_var_names_segment_bounded = [
            f"{dataset}_trend_{seg_name}_ci{rel_suffix}"
            for dataset in DATASETS.keys()
            for rel_suffix in rel_abs_suffixes
        ]

        data_vars_segment_bounded.update(
            {
                name: (
                    ("site", seg_info.dim_name, "sigma_level", "bound"),
                    np.full(
                        (
                            len(sites.index),
                            seg_info.n_segments,
                            len(SIGMA_LEVELS),
                            2,
                        ),
                        fill_value=np.nan,
                        dtype=np.float32,
                    ),  # type: ignore
                )
                for name in data_var_names_segment_bounded
            }
        )

    coords = {
        "site": sites.index.values,
        "bound": ["lower", "upper"],
        "sigma_level": SIGMA_LEVELS,
        "sigma_level_names": ("sigma_level", SIGMA_LEVEL_NAMES),
    }

    coords_segmentation = {
        seg.dim_name: seg.dim for seg in SEGMENTATIONS.values()
    }

    coords = {**coords, **coords_segmentation}  # type: ignore

    ds_output = xr.Dataset(
        data_vars={
            **data_vars_full_year,
            **data_vars_full_year_counts,
            **data_vars_full_year_bounded,
            **data_vars_segment,
            **data_vars_segment_counts,
            **data_vars_segment_bounded,
        },
        coords=coords,
    )

    return ds_output


def load_cloud_cover_data(
    datasets: dict[str, DatasetVariable],
    suffix: str,
    freq: str | None = None,
) -> dict[str, dict[str, dict[str, DataArray]]]:
    """Load data arrays for trend estimation.

    Returns:
        dict[str, dict[str, dict[str, DataArray]]]: Mapping from dataset name to
        site-level instrument series.
    """
    data_all: dict[str, dict[str, dict[str, DataArray]]] = {}
    for series_name, dataset_variable in datasets.items():
        dataset_name = dataset_variable.dataset_name
        variable_name = dataset_variable.variable_name
        data_path = (
            _level3_clouds_path(dataset_name, dataset_variable.product)
            / f"{dataset_name}_level3_clouds{'_' + freq if freq else ''}_raw{suffix}.nc"
        )
        if not data_path.exists():
            raise FileNotFoundError(
                f"Could not find {dataset_name} level-3 input for temporal granularity "
                f"'{freq}' at {data_path}."
            )
        logger.info(
            f"Loading {series_name} from {dataset_name} data at {data_path}..."
        )
        da = _load_dataarrays_from_netcdf(
            data_path,
            variable_name,
            series_name,
            dataset_name,
        )
        data_all[series_name] = da

    return data_all


def _site_granularity_groups(
    site_ids: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Group configured sites by the level-3 aggregation frequency.

    Sites without a configured frequency are deliberately omitted. Their
    output variables remain at the initialized missing value.
    """
    groups: dict[str, list[str]] = {}
    for site in site_ids:
        frequency = GRANULARITIES.get(site)
        if frequency is None:
            logger.warning(
                "Skipping site %s because no temporal granularity is configured.",
                site,
            )
            continue
        if frequency not in SUPPORTED_GRANULARITIES:
            raise ValueError(
                f"Unsupported temporal granularity '{frequency}' configured for site '{site}'. "
                f"Supported values are {sorted(SUPPORTED_GRANULARITIES)}."
            )
        groups.setdefault(frequency, []).append(site)

    return {frequency: tuple(sites) for frequency, sites in groups.items()}


def _merge_site_data_for_frequency(
    destination: dict[str, dict[str, dict[str, DataArray]]],
    source: dict[str, dict[str, dict[str, DataArray]]],
    site_ids: Iterable[str],
) -> None:
    """Merge only the requested sites from one frequency-specific load."""
    selected_sites = set(site_ids)
    for dataset_name, data_by_site in source.items():
        destination_by_site = destination.setdefault(dataset_name, {})
        for site, site_data in data_by_site.items():
            if site in selected_sites:
                destination_by_site[site] = site_data


def _candidate_cloud_cover_paths(
    dataset_name: str,
    suffix: str,
    product_stem: str,
    product: str | None = None,
) -> list[Path]:
    return [
        _level3_clouds_path(dataset_name, product)
        / f"{dataset_name}_level3_clouds_{product_stem}{suffix}.nc",
        _level3_clouds_path(dataset_name, product)
        / f"{dataset_name}_cloud_cover_{product_stem}{suffix}.nc",
    ]


def _normalize_dataarray_site_dim(da: DataArray, site_id: str) -> DataArray:
    if "site" in da.dims:
        if da.sizes.get("site", 0) == 1:
            return da.assign_coords(site=[site_id])
        if "site" in da.coords and site_id in da.site.values:
            return da.sel(site=[site_id]).assign_coords(site=[site_id])
        return da.isel(site=[0]).assign_coords(site=[site_id])

    if "site" in da.coords:
        da = da.reset_coords("site", drop=True)

    return da.expand_dims(site=[site_id])


def _select_site_instruments(
    site_data: dict[str, DataArray],
) -> dict[str, DataArray]:
    """Drop the synthetic super series when instrument-resolved series exist."""
    non_super = {
        instrument: da
        for instrument, da in site_data.items()
        if instrument != "super"
    }
    if non_super:
        return non_super
    return site_data


def _select_site_series_for_dataset(
    dataset_name: str,
    site_data: dict[str, DataArray],
) -> dict[str, DataArray] | None:
    """Select the site-level series that should feed level-4 trend estimation."""
    if dataset_name == "alc":
        return _select_site_instruments(site_data)

    return site_data


def _load_dataarray_from_datatree(
    data_path: Path,
    variable_name: str,
) -> dict[str, dict[str, DataArray]]:
    """Load site-level instrument series from a DataTree-backed file."""
    if not hasattr(xr, "open_datatree"):
        raise ValueError(
            "DataTree input detected, but this xarray version does not support open_datatree()."
        )

    dtree = xr.open_datatree(data_path)
    da_by_site_and_instrument: dict[str, dict[str, DataArray]] = {}

    for path, (node,) in group_subtrees(dtree):
        if node.children:
            continue

        parts = [part for part in path.split("/") if part]
        if len(parts) < 2:
            continue

        site_id = parts[0]
        instrument_name = parts[1]
        ds_node = node.dataset
        if variable_name not in ds_node.data_vars:
            continue

        da_by_site_and_instrument.setdefault(site_id, {})[instrument_name] = ds_node[
            variable_name
        ]

    return da_by_site_and_instrument


def _load_dataarray_from_datatree_super(
    data_path: Path,
    variable_name: str,
) -> DataArray:
    return _load_dataarray_from_datatree_group(
        data_path,
        variable_name,
        group_name="super",
    )


def _load_dataarray_from_datatree_group(
    data_path: Path,
    variable_name: str,
    group_name: str,
) -> DataArray:
    if not hasattr(xr, "open_datatree"):
        raise ValueError(
            "DataTree input detected, but this xarray version does not support open_datatree()."
        )

    dtree = xr.open_datatree(data_path)
    da_by_site: dict[str, DataArray] = {}

    for path, (node,) in group_subtrees(dtree):
        if node.children:
            continue

        parts = [part for part in path.split("/") if part]
        if len(parts) < 2 or parts[1] != group_name:
            continue

        site_id = parts[0]
        ds_node = node.dataset
        if variable_name not in ds_node.data_vars:
            continue

        da_site = _normalize_dataarray_site_dim(ds_node[variable_name], site_id)
        da_by_site[site_id] = da_site

    if not da_by_site:
        raise KeyError(
            f"Could not find variable '{variable_name}' in any '/site/{group_name}' groups in {data_path}."
        )

    return xr.concat(
        [da_by_site[site] for site in sorted(da_by_site.keys())],
        dim="site",
        join="outer",
    )


def _load_dataarrays_from_netcdf(
    data_path: Path,
    variable_name: str,
    series_name: str,
    dataset_name: str,
) -> dict[str, dict[str, DataArray]]:
    """Load site-level series while preserving instrument identity when available."""
    if dataset_name in ("era5", "ceres"):
        ds_root = xr.open_dataset(data_path)
        if variable_name not in ds_root.data_vars:
            raise KeyError(
                f"Variable '{variable_name}' not found in root dataset of {data_path}."
            )

        return {
            str(site): {
                series_name: ds_root[variable_name].sel(site=site, drop=True)
            }
            for site in ds_root.site.values
        }

    selected_by_site: dict[str, dict[str, DataArray]] = {}
    for site_id, instruments in _load_dataarray_from_datatree(
        data_path,
        variable_name,
    ).items():
        selected = _select_site_series_for_dataset(dataset_name, instruments)
        if selected is None:
            logger.warning(
                "Skipping %s site %s because the required level-4 input series is missing.",
                dataset_name,
                site_id,
            )
            continue
        selected_by_site[site_id] = selected

    return selected_by_site


def _load_dataarray_from_netcdf(
    data_path: Path,
    variable_name: str,
    group_name: str | None = None,
) -> DataArray:
    if group_name is None:
        ds_root = xr.open_dataset(data_path)
        if variable_name not in ds_root.data_vars:
            raise KeyError(
                f"Variable '{variable_name}' not found in root dataset of {data_path}."
            )
        return ds_root[variable_name]

    try:
        ds_root = xr.open_dataset(data_path)
        if variable_name in ds_root.data_vars:
            return ds_root[variable_name]
    except Exception:
        logger.debug(
            f"Root dataset read failed for {data_path}, trying DataTree '/site/{group_name}' groups."
        )

    return _load_dataarray_from_datatree_group(
        data_path,
        variable_name,
        group_name=group_name,
    )


def _climatology_group_name(dataset_name: str) -> str | None:
    if dataset_name == "alc":
        return "shared_climatology"
    return None

def load_climatology_data(
    datasets: dict[str, DatasetVariable],
    suffix: str,
    segmentations: dict[str, Segmentation],
) -> dict[str, dict[str, DataArray]]:
    climatologies_all: dict[str, dict[str, DataArray]] = {}
    for series_name, dataset_variable in datasets.items():
        dataset_name = dataset_variable.dataset_name
        variable_name = dataset_variable.variable_name
        candidates = _candidate_cloud_cover_paths(
            dataset_name,
            suffix,
            product_stem="mean_climatology",
            product=dataset_variable.product,
        )
        data_path = next((path for path in candidates if path.exists()), None)
        if data_path is None:
            raise FileNotFoundError(
                f"Could not find mean climatology file for '{dataset_name}'. Checked: {candidates}"
            )

        logger.info(
            f"Loading {series_name} climatology from {dataset_name} data at {data_path}..."
        )
        da: DataArray = _load_dataarray_from_netcdf(
            data_path,
            variable_name,
            group_name=_climatology_group_name(dataset_name),
        )
        climatologies_all[series_name] = {}
        climatologies_all[series_name]["mean"] = da

        for seg_name, seg_info in segmentations.items():
            candidates_seg = _candidate_cloud_cover_paths(
                dataset_name,
                suffix,
                product_stem=f"{seg_info.climatology}_mean_climatology",
                product=dataset_variable.product,
            )
            climatology_path = next(
                (path for path in candidates_seg if path.exists()),
                None,
            )
            if climatology_path is None:
                raise FileNotFoundError(
                    f"Could not find {seg_info.climatology} climatology file for '{dataset_name}'. Checked: {candidates_seg}"
                )

            logger.info(
                f"Loading {series_name} {seg_info.climatology} climatology from {dataset_name} data at {climatology_path}..."
            )
            da_seg: DataArray = _load_dataarray_from_netcdf(
                climatology_path,
                variable_name,
                group_name=_climatology_group_name(dataset_name),
            )
            climatologies_all[series_name][seg_name] = da_seg

    return climatologies_all


def _site_ids(sites: GeoDataFrame | Iterable[str]) -> tuple[str, ...]:
    """Normalize site containers to an ordered tuple of site identifiers."""
    index = getattr(sites, "index", None)
    if index is not None and not callable(index):
        return tuple(str(site) for site in index)
    return tuple(str(site) for site in sites)


def check_data_coverage(
    da_reference: dict[str, dict[str, DataArray]],
    sites: GeoDataFrame | Iterable[str],
    min_periods: int | None = None,
    min_years: int | None = None,
) -> dict[str, dict[str, bool | int | np.datetime64 | None]]:
    """Check the data coverages in the reference dataset.

    Args:
        da_reference (DataArray): DataArray of the reference dataset.
        sites (GeoDataFrame | Iterable[str]): Ground sites or their identifiers.
        min_periods (int, optional): Minimum number of valid aggregation periods required.
        min_years (int, optional): Minimum calendar span after applying the record
            cutoffs. This is independent of the input aggregation frequency.

    Returns:
        dict[str, dict[str, bool | np.datetime64 | None]]: Mapping from site to temporal coverage info.
    """
    if min_periods is None and min_years is None:
        raise ValueError("Either min_periods or min_years must be provided.")
    if min_periods is not None and min_years is not None:
        raise ValueError("Specify only one of min_periods or min_years.")

    data_coverage: dict[str, dict[str, bool | int | np.datetime64 | None]] = {}
    dummy: dict[str, bool | int | np.datetime64 | None] = {
        "has_enough_data": False,
        "num_valid_days": None,
        "first_valid_date": None,
        "last_valid_date": None,
        "effective_start_date": None,
        "effective_end_date": None,
    }

    for site in _site_ids(sites):
        if site not in da_reference:
            data_coverage[site] = dummy
            continue

        da_site = _select_site_instruments(da_reference[site])
        da_valid = xr.concat(
            [da.where(da.notnull(), drop=True) for da in da_site.values()],
            dim="time",
        ).sortby("time")
        t_first = da_valid.time.isel(time=0).values
        t_last = da_valid.time.isel(time=-1).values
        effective_start = max(t_first, RECORD_CUTOFF_DATE_START)
        effective_end = min(t_last, RECORD_CUTOFF_DATE_END)
        if min_periods is not None and da_valid.time.size < min_periods:
            data_coverage[site] = dummy
            continue
        if min_years is not None and effective_end < effective_start + np.timedelta64(
            365 * min_years, "D"
        ):
            data_coverage[site] = dummy
            continue

        data_coverage[site] = {
            "has_enough_data": True,
            "num_valid_days": int(da_valid.time.size),
            "first_valid_date": t_first,
            "last_valid_date": t_last,
            "effective_start_date": effective_start,
            "effective_end_date": effective_end,
        }

    return data_coverage


def mask_valid_data_coverage(
    data_all: dict[str, dict[str, dict[str, DataArray]]],
    data_coverage: dict[str, dict[str, bool | int | np.datetime64 | None]],
) -> dict[str, dict[str, dict[str, DataArray]]]:
    """Mask data arrays to the maximum reference data coverage period for a given site.

    Args:
        data_all (dict[str, dict[str, dict[str, DataArray]]]): Mapping from dataset
            name to site-level instrument series.
        data_coverage (dict[str, dict[str, bool | int | np.datetime64 | None]]): Data coverage information for each site.

    Returns:
        dict[str, dict[str, dict[str, DataArray]]]: Sliced data arrays for each site and instrument.
    """
    for ds_name, ds_by_site in data_all.items():
        for site, instrument_data in ds_by_site.items():
            coverage_info = data_coverage.get(site, {})
            if not coverage_info.get("has_enough_data", False):
                continue
            t_start = coverage_info["effective_start_date"]
            t_end = coverage_info["effective_end_date"]
            for instrument, da in instrument_data.items():
                data_all[ds_name][site][instrument] = da.where(
                    (da.time >= t_start) & (da.time <= t_end)
                )

    return data_all


def _sorted_valid_days(da: DataArray) -> np.ndarray:
    """Return sorted unique valid calendar days for one instrument series."""
    da_sorted = da.sortby("time")
    time_values = np.asarray(da_sorted.time.values)
    value_values = np.asarray(da_sorted.values, dtype=np.float64)
    valid_mask = ~np.isnan(value_values)
    if not np.any(valid_mask):
        return np.array([], dtype="datetime64[D]")

    valid_days = np.asarray(time_values[valid_mask]).astype("datetime64[D]")
    return np.unique(valid_days)

def _remove_overlapping_alc_instrument_days(
    site_data: dict[str, DataArray],
    overlap_window_days: int = ALC_OVERLAP_WINDOW_DAYS,
) -> dict[str, DataArray]:
    """Assign each ALC timestamp to one instrument by coverage priority.

    Priority is deterministic: most valid days first, then earliest first valid
    day, then instrument name. Timestamps within +/- ``overlap_window_days``
    are treated as overlapping.
    """
    selected_site_data = _select_site_instruments(site_data)
    if len(selected_site_data) <= 1:
        return selected_site_data

    instrument_stats: list[tuple[str, np.ndarray]] = []
    for instrument, da in selected_site_data.items():
        valid_days = _sorted_valid_days(da)
        instrument_stats.append((instrument, valid_days))

    instrument_stats.sort(
        key=lambda item: (
            -item[1].size,
            item[1][0] if item[1].size > 0 else np.datetime64("NaT"),
            item[0],
        )
    )

    claimed_day_windows: set[int] = set()
    filtered_site_data: dict[str, DataArray] = {}

    for instrument, _valid_days in instrument_stats:
        da_sorted = selected_site_data[instrument].sortby("time")
        sample_day_keys = np.asarray(da_sorted.time.values).astype(
            "datetime64[D]"
        ).astype(np.int64)
        keep_sample_mask = np.array(
            [day_key not in claimed_day_windows for day_key in sample_day_keys],
            dtype=bool,
        )

        da_filtered = da_sorted.isel(
            time=np.flatnonzero(keep_sample_mask)
        )
        filtered_site_data[instrument] = da_filtered

        kept_day_keys = _sorted_valid_days(da_filtered).astype(np.int64)
        for kept_day_key in kept_day_keys.tolist():
            start = kept_day_key - overlap_window_days
            end = kept_day_key + overlap_window_days
            claimed_day_windows.update(range(start, end + 1))

    return filtered_site_data


def _combine_site_instrument_series(
    dataset_name: str,
    site_data: dict[str, DataArray],
    granularity: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine instrument-specific series into aligned arrays for fitting."""
    if dataset_name == "alc" and EXCLUDE_OVERLAP:
        overlap_window_days = ALC_OVERLAP_WINDOW_DAYS
        if granularity is not None:
            try:
                overlap_window_days = ALC_OVERLAP_WINDOW_DAYS_BY_GRANULARITY[
                    granularity
                ]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported temporal granularity '{granularity}' for ALC overlap handling."
                ) from exc
        site_data_for_fit = _remove_overlapping_alc_instrument_days(
            site_data,
            overlap_window_days=overlap_window_days,
        )
    elif dataset_name == "alc":
        site_data_for_fit = _select_site_instruments(site_data)
    else:
        site_data_for_fit = _select_site_instruments(site_data)

    time_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    instrument_parts: list[np.ndarray] = []

    for instrument, da in site_data_for_fit.items():
        da_sorted = da.sortby("time")
        time_parts.append(np.asarray(da_sorted.time.values))
        value_parts.append(np.asarray(da_sorted.values, dtype=np.float64))
        instrument_parts.append(
            np.full(da_sorted.time.size, instrument, dtype=object)
        )

    if not time_parts:
        empty = np.array([], dtype=np.float64)
        return empty.astype("datetime64[ns]"), empty, empty.astype(object)

    time_values = np.concatenate(time_parts)
    value_values = np.concatenate(value_parts)
    instrument_values = np.concatenate(instrument_parts)
    sort_index = np.argsort(time_values)

    return (
        time_values[sort_index],
        value_values[sort_index],
        instrument_values[sort_index],
    )


def _count_valid_days_for_trend(
    time_values: np.ndarray,
    value_values: np.ndarray,
) -> int:
    valid_mask = ~np.isnan(value_values)
    if not np.any(valid_mask):
        return 0

    valid_days = np.asarray(time_values[valid_mask]).astype("datetime64[D]")
    return int(np.unique(valid_days).size)


def _estimate_site_trends_for_dataset(
    dataset_name: str,
    site: str,
    site_data: dict[str, DataArray] | None,
    has_enough_data: bool,
    granularity: str | None = None,
) -> tuple[
    list[tuple[str, dict[str, object], float]],
    TrendSlopeQuantileRecord | None,
    list[SeasonalTrendSlopeQuantileRecord],
]:
    """Estimate full-year and segmented trends for one site/dataset pair."""
    if site_data is None:
        logger.debug(f" Site {site} not found in data array, skipping...")
        return [], None, []

    if not has_enough_data:
        logger.debug(
            f" Site {site} does not have enough {REFERENCE_DATASET} data, skipping..."
        )
        return [], None, []

    logger.info(f" Estimating trends for {dataset_name} from site {site}.")
    updates: list[tuple[str, dict[str, object], float]] = []

    site_time, site_values, site_instruments = _combine_site_instrument_series(
        dataset_name,
        site_data,
        granularity=granularity,
    )
    if site_values.size == 0:
        logger.debug(
            f" Site {site} does not contain any instrument series for {dataset_name}, skipping..."
        )
        return [], None, []

    n_samples = _count_valid_days_for_trend(site_time, site_values)
    updates.append(
        (
            f"{dataset_name}_trend_n_samples",
            {"site": site},
            float(n_samples),
        )
    )

    site_time_da = xr.DataArray(site_time, dims="time")

    full_year_estimator = MannKendallTrendEstimator(
        resolution=1e-12,
        pw_method="3pw",
        alpha_ak=95.,
        # Keep estimator-internal slope generation serial. The level-4 entrypoint
        # parallelizes over suffix-site tasks instead.
        same_instrument_only=(dataset_name == "alc") and SAME_INSTRUMENT_ONLY_FOR_ALC,
        same_season_only=True,
        n_jobs=1,
    )
    try:
        full_year_estimator.fit(
            site_time,
            site_values,
            instrument_labels=site_instruments,
        )
    except Exception as exc:
        logger.warning(
            " Skipping %s site %s because full-year trend could not be estimated: %s: %s",
            dataset_name,
            site,
            type(exc).__name__,
            exc,
        )
        return [], None, []

    pairwise_slope_record: TrendSlopeQuantileRecord | None = None
    try:
        trend_slope_quantile_function = (
            full_year_estimator.get_trend_slope_quantile_function(
                n_alpha_points=TREND_SLOPE_QUANTILE_ALPHA_POINTS,
            )
        )
        pairwise_slope_record = TrendSlopeQuantileRecord(
            dataset_name=dataset_name,
            site=site,
            probability_levels=trend_slope_quantile_function["probability_levels"],
            quantile_values=trend_slope_quantile_function["quantile_values"],
            n_slopes=trend_slope_quantile_function["n_slopes"],
            kendall_variance=trend_slope_quantile_function["kendall_variance"],
            slope=trend_slope_quantile_function["slope"],
            series_key=trend_slope_quantile_function["series_key"],
            n_samples=int(n_samples),
        )
    except Exception as exc:
        logger.warning(
            " Could not build trend-slope quantile function for %s site %s: %s: %s",
            dataset_name,
            site,
            type(exc).__name__,
            exc,
        )

    segment_slope_records: list[SeasonalTrendSlopeQuantileRecord] = []

    site_segmented_vals = {
        seg_name: seg_info.segmentation_func(site_time_da, site_values)
        for seg_name, seg_info in SEGMENTATIONS.items()
    }
    site_segmented_dts = {
        seg_name: seg_info.segmentation_func(site_time_da, site_time)
        for seg_name, seg_info in SEGMENTATIONS.items()
    }
    site_segmented_instruments = {
        seg_name: seg_info.segmentation_func(site_time_da, site_instruments)
        for seg_name, seg_info in SEGMENTATIONS.items()
    }
    site_segmented_n_samples = {
        seg_name: {
            segment: _count_valid_days_for_trend(
                site_segmented_dts[seg_name][segment],
                site_segmented_vals[seg_name][segment],
            )
            for segment in seg_info.dim
        }
        for seg_name, seg_info in SEGMENTATIONS.items()
    }

    for seg_name, seg_info in SEGMENTATIONS.items():
        for segment in seg_info.dim:
            updates.append(
                (
                    f"{dataset_name}_trend_{seg_name}_n_samples",
                    {"site": site, seg_info.dim_name: segment},
                    float(site_segmented_n_samples[seg_name][segment]),
                )
            )

    for sigma_level in SIGMA_LEVELS:
        mk_out_full_year = full_year_estimator.get_result(
            alpha_cl=sigma_level * 100.0
        )

        ci_name = f"{dataset_name}_trend_ci"
        updates.append(
            (
                ci_name,
                {
                    "site": site,
                    "sigma_level": sigma_level,
                    "bound": "lower",
                },
                float(mk_out_full_year["lcl"]),
            )
        )
        updates.append(
            (
                ci_name,
                {
                    "site": site,
                    "sigma_level": sigma_level,
                    "bound": "upper",
                },
                float(mk_out_full_year["ucl"]),
            )
        )

        if round(sigma_level - 0.6827, 2) == 0:
            updates.append(
                (
                    f"{dataset_name}_trend",
                    {"site": site},
                    float(mk_out_full_year["slope"]),
                )
            )
            updates.append(
                (
                    f"{dataset_name}_trend_pvalue",
                    {"site": site},
                    float(mk_out_full_year["p"]),
                )
            )
            updates.append(
                (
                    f"{dataset_name}_trend_std",
                    {"site": site},
                    float(mk_out_full_year["ucl"] - mk_out_full_year["lcl"]) / 2.0,
                )
            )

        for seg_name, seg_info in SEGMENTATIONS.items():
            for segment in seg_info.dim:
                if site_segmented_n_samples[seg_name][segment] <= 10:
                    continue

                segment_estimator = MannKendallTrendEstimator(
                    resolution=1e-12,
                    pw_method="3pw",
                    alpha_ak=95.,
                    # Keep estimator-internal slope generation serial. The level-4 entrypoint
                    # parallelizes over suffix-site tasks instead.
                    same_instrument_only=(dataset_name == "alc") and SAME_INSTRUMENT_ONLY_FOR_ALC,
                    same_season_only=(seg_name == "by_season"),
                    n_jobs=1,
                )
                try:
                    segment_estimator.fit(
                        site_segmented_dts[seg_name][segment],
                        site_segmented_vals[seg_name][segment].astype(np.float32),
                        instrument_labels=site_segmented_instruments[seg_name][segment],
                    )
                    mk_out_segment = segment_estimator.get_result(
                        alpha_cl=sigma_level * 100.0
                    )
                except Exception as exc:
                    logger.warning(
                        " Skipping %s %s segment %s for site %s because the trend could not be estimated: %s: %s",
                        dataset_name,
                        seg_name,
                        segment,
                        site,
                        type(exc).__name__,
                        exc,
                    )
                    continue

                ci_name_segment = f"{dataset_name}_trend_{seg_name}_ci"
                updates.append(
                    (
                        ci_name_segment,
                        {
                            "site": site,
                            seg_info.dim_name: segment,
                            "sigma_level": sigma_level,
                            "bound": "lower",
                        },
                        float(mk_out_segment["lcl"]),
                    )
                )
                updates.append(
                    (
                        ci_name_segment,
                        {
                            "site": site,
                            seg_info.dim_name: segment,
                            "sigma_level": sigma_level,
                            "bound": "upper",
                        },
                        float(mk_out_segment["ucl"]),
                    )
                )

                if round(sigma_level - 0.6827, 2) == 0:
                    updates.append(
                        (
                            f"{dataset_name}_trend_{seg_name}",
                            {"site": site, seg_info.dim_name: segment},
                            float(mk_out_segment["slope"]),
                        )
                    )
                    updates.append(
                        (
                            f"{dataset_name}_trend_{seg_name}_pvalue",
                            {"site": site, seg_info.dim_name: segment},
                            float(mk_out_segment["p"]),
                        )
                    )
                    updates.append(
                        (
                            f"{dataset_name}_trend_{seg_name}_std",
                            {"site": site, seg_info.dim_name: segment},
                            float(mk_out_segment["ucl"] - mk_out_segment["lcl"]) / 2.0,
                        )
                    )

                    if seg_name in SEASONAL_TREND_SLOPE_QUANTILE_SEGMENTATIONS:
                        try:
                            segment_quantile_function = (
                                segment_estimator.get_trend_slope_quantile_function(
                                    n_alpha_points=TREND_SLOPE_QUANTILE_ALPHA_POINTS,
                                )
                            )
                            segment_slope_records.append(
                                SeasonalTrendSlopeQuantileRecord(
                                    dataset_name=dataset_name,
                                    site=site,
                                    season=str(segment),
                                    probability_levels=segment_quantile_function[
                                        "probability_levels"
                                    ],
                                    quantile_values=segment_quantile_function[
                                        "quantile_values"
                                    ],
                                    n_slopes=segment_quantile_function["n_slopes"],
                                    kendall_variance=segment_quantile_function[
                                        "kendall_variance"
                                    ],
                                    slope=segment_quantile_function["slope"],
                                    series_key=segment_quantile_function["series_key"],
                                    n_samples=int(
                                        site_segmented_n_samples[seg_name][segment]
                                    ),
                                )
                            )
                        except Exception as exc:
                            logger.warning(
                                " Could not build trend-slope quantile function for %s %s segment %s site %s: %s: %s",
                                dataset_name,
                                seg_name,
                                segment,
                                site,
                                type(exc).__name__,
                                exc,
                            )

    return updates, pairwise_slope_record, segment_slope_records


def _compute_suffix_updates(
    task: SuffixTask,
) -> tuple[
    str,
    list[tuple[str, dict[str, object], float]],
    list[TrendSlopeQuantileRecord],
    list[SeasonalTrendSlopeQuantileRecord],
]:
    """Estimate all dataset updates for one suffix in a worker process."""
    logger.info(f"Preparing worker inputs for suffix: '{task.suffix}'")
    datasets: dict[str, dict[str, dict[str, DataArray]]] = {}
    configured_site_groups = _site_granularity_groups(task.site_ids)
    for frequency, site_ids in configured_site_groups.items():
        try:
            frequency_data = load_cloud_cover_data(
                DATASETS,
                suffix=task.suffix,
                freq=frequency,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not load temporal granularity '{frequency}' for sites "
                f"{list(site_ids)}: {exc}"
            ) from exc
        _merge_site_data_for_frequency(datasets, frequency_data, site_ids)

    data_coverage_daily = check_data_coverage(
        datasets.get(REFERENCE_DATASET, {}),
        task.site_ids,
        min_years=MIN_YEARS,
    )
    datasets = mask_valid_data_coverage(datasets, data_coverage_daily)

    updates: list[tuple[str, dict[str, object], float]] = []
    pairwise_slope_records: list[TrendSlopeQuantileRecord] = []
    seasonal_slope_records: list[SeasonalTrendSlopeQuantileRecord] = []
    for site in task.site_ids:
        site_granularity = GRANULARITIES.get(site)
        has_enough_data = bool(
            data_coverage_daily.get(site, {}).get("has_enough_data", False)
        )
        for dataset_name, da_by_site in datasets.items():
            site_updates, pairwise_slope_record, segment_slope_records = (
                _estimate_site_trends_for_dataset(
                    dataset_name,
                    site,
                    da_by_site.get(site),
                    has_enough_data,
                    granularity=site_granularity,
                )
            )
            updates.extend(site_updates)
            if pairwise_slope_record is not None:
                pairwise_slope_records.append(pairwise_slope_record)
            seasonal_slope_records.extend(segment_slope_records)
    return task.suffix, updates, pairwise_slope_records, seasonal_slope_records


def _build_suffix_tasks(
    sites: GeoDataFrame,
) -> list[SuffixTask]:
    """Build one light-weight multiprocessing task per suffix."""
    site_ids = _site_ids(sites)
    unknown_configured_sites = set(GRANULARITIES) - set(site_ids)
    if unknown_configured_sites:
        raise ValueError(
            "Temporal granularity is configured for sites absent from the ground-site metadata: "
            f"{sorted(unknown_configured_sites)}"
        )
    return [SuffixTask(suffix=suffix, site_ids=site_ids) for suffix in SUFFIXES]


def _collect_suffix_updates(
    tasks: list[SuffixTask],
) -> tuple[
    dict[str, list[tuple[str, dict[str, object], float]]],
    dict[str, list[TrendSlopeQuantileRecord]],
    dict[str, list[SeasonalTrendSlopeQuantileRecord]],
]:
    """Run one multiprocessing worker per suffix and collect their updates."""
    task_results = parallel_map(
        _compute_suffix_updates,
        tasks,
        n_jobs=min(N_JOBS, max(1, len(tasks))),
        batch_size=max(1, len(tasks)),
    )

    updates_by_suffix: dict[str, list[tuple[str, dict[str, object], float]]] = {
        suffix: [] for suffix in SUFFIXES
    }
    pairwise_slope_records_by_suffix: dict[str, list[TrendSlopeQuantileRecord]] = {
        suffix: [] for suffix in SUFFIXES
    }
    seasonal_slope_records_by_suffix: dict[
        str, list[SeasonalTrendSlopeQuantileRecord]
    ] = {suffix: [] for suffix in SUFFIXES}
    for (
        suffix,
        updates,
        pairwise_slope_records,
        seasonal_slope_records,
    ) in task_results:
        updates_by_suffix[suffix] = updates
        pairwise_slope_records_by_suffix[suffix] = pairwise_slope_records
        seasonal_slope_records_by_suffix[suffix] = seasonal_slope_records
    return (
        updates_by_suffix,
        pairwise_slope_records_by_suffix,
        seasonal_slope_records_by_suffix,
    )


def _write_pairwise_slope_parquet(
    records: list[TrendSlopeQuantileRecord],
    suffix: str,
    output_dir: Path,
) -> None:
    """Write one tidy parquet file per dataset-site pair with the trend-slope
    quantile function (i.e. the MK-test-inverted sampling distribution of
    the trend estimate, not the empirical distribution of pairwise slopes).
    """
    for record in records:
        frame = pd.DataFrame(
            {
                "probability_level": record.probability_levels,
                "quantile_value_per_year": record.quantile_values,
            }
        )
        frame["dataset"] = record.dataset_name
        frame["site"] = record.site
        frame["suffix"] = suffix
        frame["series_key"] = record.series_key
        frame["slope_per_year"] = record.slope
        frame["kendall_variance"] = record.kendall_variance
        frame["n_slopes"] = record.n_slopes
        frame["n_samples"] = record.n_samples

        output_path = (
            output_dir
            / f"{record.dataset_name}_{record.site}_trend_slope_distribution{suffix}.parquet"
        )
        frame.to_parquet(output_path, index=False)


def _write_seasonal_trend_slope_quantile_parquet(
    records: list[SeasonalTrendSlopeQuantileRecord],
    suffix: str,
    output_dir: Path,
) -> None:
    """Write one tidy parquet file per dataset-site-season combination with
    the seasonal trend-slope quantile function (i.e. the MK-test-inverted
    sampling distribution of the seasonal trend estimate, not the empirical
    distribution of pairwise slopes).
    """
    for record in records:
        frame = pd.DataFrame(
            {
                "probability_level": record.probability_levels,
                "quantile_value_per_year": record.quantile_values,
            }
        )
        frame["dataset"] = record.dataset_name
        frame["site"] = record.site
        frame["season"] = record.season
        frame["suffix"] = suffix
        frame["series_key"] = record.series_key
        frame["slope_per_year"] = record.slope
        frame["kendall_variance"] = record.kendall_variance
        frame["n_slopes"] = record.n_slopes
        frame["n_samples"] = record.n_samples

        output_path = (
            output_dir
            / f"{record.dataset_name}_{record.site}_{record.season}_trend_slope_distribution{suffix}.parquet"
        )
        frame.to_parquet(output_path, index=False)


def _apply_updates_to_output(
    ds_output: Dataset,
    updates: list[tuple[str, dict[str, object], float]],
) -> Dataset:
    """Apply worker updates to the suffix output dataset on the coordinator thread."""
    for var_name, indexer, value in updates:
        ds_output[var_name].loc[indexer] = value
    return ds_output


def mk_segmented_trend_estimation(
    ds_output: Dataset,
    updates: list[tuple[str, dict[str, object], float]],
) -> Dataset:
    """Apply full-year and segmented trend updates to an output dataset.

    Args:
        ds_output (Dataset): Dataset to store the output trends.
        updates (list[tuple[str, dict[str, object], float]]): Flat list of
            worker-produced updates to merge into the dataset.

    Returns:
        Dataset: Dataset with estimated trends.
    """
    return _apply_updates_to_output(ds_output, updates)



def compute_relative_values(
    ds_output: Dataset,
    climatologies: dict[str, dict[str, DataArray]],
) -> Dataset:
    """Compute relative (normalized) values of trends.

    Normalizes absolute trends by dividing by the climatological mean.
    This converts absolute trends (e.g., fraction/year) to relative trends (e.g., %/year).

    N.B. The current implementation is very ugly. The loops over sites could be avoided by e.g.
    mutual reindexing of site coordinates.

    Args:
        ds_output (Dataset): Dataset containing absolute trends.
        ds_mean_climatology (dict[str, DataArray]): DataArrays with overall mean climatology, indexed by site.
        ds_mean_monthly_climatology (dict[str, DataArray]): DataArrays with monthly mean climatology, indexed by (site, month).
        ground_sites (GeoDataFrame): GeoDataFrame containing ground site information.

    Returns:
        Dataset: Updated dataset with relative (_rel) variables populated.
    """

    # Normalize full year trends by overall climatology
    for dataset_name in DATASETS.keys():
        logger.info(
            f"Computing relative full year trends for {dataset_name}..."
        )

        # Get mean climatology data - this should be a DataArray indexed by site
        if dataset_name in climatologies:
            climatology = climatologies[dataset_name]["mean"]
        else:
            logger.debug(
                f" Climatology not available for {dataset_name}, skipping relative normalization."
            )
            continue
        for site in ds_output.site.values:
            if site not in climatology.site.values:
                logger.debug(
                    f" Site {site} not found in climatology for {dataset_name}, skipping relative normalization."
                )
                continue

            climatology_site = climatology.sel(site=site)
            ds_site = ds_output.sel(site=site)

            # Normalize full year trend
            trend_abs = ds_site[f"{dataset_name}_trend"]
            trend_rel = trend_abs / climatology_site
            ds_output[f"{dataset_name}_trend_relative"].loc[
                dict(site=site)
            ] = trend_rel

            # Normalize full year trend standard deviation using error propagation
            # For z = x/y, σ_z = |z| * sqrt((σ_x/x)^2 + (σ_y/y)^2)
            # Assuming climatology has negligible uncertainty, simplifies to σ_z = σ_x / |y|
            trend_std_abs = ds_site[f"{dataset_name}_trend_std"]
            trend_std_rel = trend_std_abs / np.abs(climatology_site)
            ds_output[f"{dataset_name}_trend_std_relative"].loc[
                dict(site=site)
            ] = trend_std_rel

            # Normalize full year confidence intervals
            ci_abs = ds_site[f"{dataset_name}_trend_ci"]
            ci_rel = ci_abs / climatology_site
            ds_output[f"{dataset_name}_trend_ci_relative"].loc[
                dict(site=site)
            ] = ci_rel

        # Normalize segmented trends by corresponding climatology
        for seg_name in SEGMENTATIONS.keys():
            if dataset_name in climatologies:
                climatology = climatologies[dataset_name][seg_name]
            else:
                logger.debug(
                    f" Monthly climatology not available for {dataset_name}, skipping relative normalization."
                )
                continue
            for site in ds_output.site.values:
                if site not in climatology.site.values:
                    logger.debug(
                        f" Site {site} not found in monthly climatology for {dataset_name}, skipping relative normalization."
                    )
                    continue

                climatology_site = climatology.sel(site=site)
                ds_site = ds_output.sel(site=site)

                # Normalize monthly trend
                trend_abs = ds_site[f"{dataset_name}_trend_{seg_name}"]
                trend_rel = trend_abs / climatology_site
                ds_output[f"{dataset_name}_trend_{seg_name}_relative"].loc[
                    dict(site=site)
                ] = trend_rel

                # Normalize monthly trend standard deviation
                trend_std_abs = ds_site[f"{dataset_name}_trend_{seg_name}_std"]
                trend_std_rel = trend_std_abs / np.abs(climatology_site)
                ds_output[f"{dataset_name}_trend_{seg_name}_std_relative"].loc[
                    dict(site=site)
                ] = trend_std_rel

                # Normalize monthly confidence intervals
                ci_abs = ds_site[f"{dataset_name}_trend_{seg_name}_ci"]
                ci_rel = ci_abs / climatology_site
                ds_output[f"{dataset_name}_trend_{seg_name}_ci_relative"].loc[
                    dict(site=site)
                ] = ci_rel

    logger.info("Completed computing relative values.")
    return ds_output


def run() -> None:
    """Main processing pipeline for trend estimation from ALC, ERA5 and CERES."""

    logger.info("Reading site metadata...")
    ground_sites = get_ground_sites_gdf()

    #ground_sites = ground_sites[
    #    ground_sites.index.isin(
    #    ["nsa"]
    #    )
    #]

    suffix_tasks = _build_suffix_tasks(ground_sites)
    (
        suffix_updates,
        suffix_pairwise_slope_records,
        suffix_seasonal_slope_records,
    ) = _collect_suffix_updates(suffix_tasks)

    out_path = DATA_PATHS.level4_trends
    out_path.mkdir(parents=True, exist_ok=True)

    trend_slope_distributions_out_path = DATA_PATHS.level4_trends_slope_distributions
    trend_slope_distributions_out_path.mkdir(parents=True, exist_ok=True)

    for suffix in SUFFIXES:
        logger.info(f"Finalizing suffix: '{suffix}'")
        ds_output = initialize_output_dataset(ground_sites)
        ds_output = mk_segmented_trend_estimation(
            ds_output,
            suffix_updates[suffix],
        )
        climatologies = load_climatology_data(DATASETS, suffix, SEGMENTATIONS)
        ds_output = compute_relative_values(
            ds_output,
            climatologies,
        )
        ds_output.to_netcdf(DATA_PATHS.level4_trends_dataset(suffix))

        _write_pairwise_slope_parquet(
            suffix_pairwise_slope_records[suffix],
            suffix,
            trend_slope_distributions_out_path,
        )
        _write_seasonal_trend_slope_quantile_parquet(
            suffix_seasonal_slope_records[suffix],
            suffix,
            trend_slope_distributions_out_path,
        )


if __name__ == "__main__":
    setup_logging(logging.INFO)

    run()