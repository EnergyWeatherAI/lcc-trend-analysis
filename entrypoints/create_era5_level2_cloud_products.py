import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from dotenv import load_dotenv

from lcc_trend_analysis.logging import (
    get_logger,
    setup_logging,
)
from lcc_trend_analysis.paths import get_data_paths

logger: logging.Logger = get_logger(__name__)

load_dotenv()

DATA_PATHS = get_data_paths()

ERA5_DATA_PATH: Path = DATA_PATHS.era5_n320
OUTPUT_DATA_PATH: Path = DATA_PATHS.era5_level2_clouds
N_JOBS: int = int(os.environ.get("N_JOBS", multiprocessing.cpu_count()))


# Geometric surface heights from ERA5
GRAVITATIONAL_ACCELERATION = (
    9.80665  # m/s^2, gravitational acceleration as in IFS
)

LOWER_CLOUD_HEIGHT_THRESHOLD = 0.0  # m above surface
ISCCP_LOW_CLOUD_TOP_PRESSURE_THRESHOLD = 68_000.0  # Pa


def cloud_base_height_agl(
    cloud_base_height_asl: xr.DataArray,
    surface_altitude: xr.DataArray,
) -> xr.DataArray:
    """Convert cloud-base height above sea level to above-ground level."""

    return cloud_base_height_asl - surface_altitude


def compute_model_level_pressure(ds: xr.Dataset) -> xr.DataArray:
    """Compute pressure on ERA5 full model levels from lnsp and hybrid coefficients."""
    if "lnsp" not in ds:
        raise KeyError(
            "Dataset is missing lnsp required for model-level pressure."
        )
    if "pv" not in ds:
        raise KeyError(
            "Dataset is missing pv required for model-level pressure."
        )

    surface_pressure = xr.apply_ufunc(np.exp, ds["lnsp"], dask="parallelized")

    pv_da = ds["pv"]
    for dim in pv_da.dims:
        if dim != "pv_index":
            pv_da = pv_da.isel({dim: 0}, drop=True)

    pv = np.asarray(pv_da.values)
    if pv.ndim != 1:
        raise ValueError(
            f"Expected pv to be 1D after dropping non-pv dimensions, got shape {pv.shape}"
        )

    nlev_full = (len(pv) // 2) - 1
    a_full = pv[: nlev_full + 1]
    b_full = pv[nlev_full + 1 :]

    hybrid_values = ds["hybrid"].values.astype(int)
    hybrid_values_sorted = np.sort(hybrid_values)
    a_half = a_full[hybrid_values_sorted[0] - 1 : hybrid_values_sorted[-1] + 2]
    b_half = b_full[hybrid_values_sorted[0] - 1 : hybrid_values_sorted[-1] + 2]

    a_da = xr.DataArray(a_half, dims=("hybrid_half",))
    b_da = xr.DataArray(b_half, dims=("hybrid_half",))
    p_half = a_da + b_da * surface_pressure
    p_full = 0.5 * (
        p_half.isel(hybrid_half=slice(0, -1))
        + p_half.isel(hybrid_half=slice(1, None))
    )
    p_full = p_full.rename({"hybrid_half": "hybrid"})
    p_full = p_full.assign_coords(hybrid=hybrid_values_sorted)
    p_full = p_full.reindex(hybrid=hybrid_values)
    p_full = p_full.transpose(*ds["cc"].dims)

    p_full.attrs["long_name"] = "Pressure on ERA5 model full levels"
    p_full.attrs["standard_name"] = "air_pressure"
    p_full.attrs["units"] = "Pa"
    p_full.attrs["positive"] = "down"

    return p_full


def get_cloud_layer_overlap_decorrelation_length(
    latitude_degrees: float,
) -> float:
    """Decorrelation length for exponential-random cloud overlap model (z0)

    Source: IFS Documentation - Cy49r1. Part IV - Physical Processes. Chapter 2: Radiation, Section 2.4.2 (c) "Cloud overlap"

    Args:
        latitude_degrees (float): Latitude in degrees.

    Returns:
        float: Decorrelation length (m)
    """
    return (0.75 + 2.149 * np.cos(np.deg2rad(latitude_degrees)) ** 2.0) * 1e3


def exponential_random_cloud_overlap(
    cc_current_level: float,
    cc_prev_levels: float,
    height_current_level: float,
    height_prev_level: float,
    z0: float,
) -> float:
    """Stochastic exponential-random cloud overlap model.

    Based on Räisänen et al (2004): Stochastic generation of subgrid-scale cloudy columns for
    large-scale models. Q.J.R. Meteorol. Soc., 130: 2047-2067. https://doi.org/10.1256/qj.03.99

    Args:
        z0 (float): Decorrelation length (m)
        cc_current_level (float): Cloud cover fraction of the current level (-)
        cc_prev_levels (float): Vertically projected cloud cover fraction of the previous levels (-)
        height_current_level (float): Geometric height of the current level (m)
        height_prev_level (float): Geometric height of the previous level (-)
    Returns:
        float: Vertically projected cloud cover fraction up to current level (-)
    """
    delta_z: float = abs(height_current_level - height_prev_level)
    cc_max: float = max(cc_current_level, cc_prev_levels)
    cc_random: float = (
        cc_current_level + cc_prev_levels - cc_prev_levels * cc_current_level
    )
    alpha: float = np.exp(-delta_z / z0)
    cc_new: float = alpha * cc_max + (1 - alpha) * cc_random
    return cc_new


def compute_cumulative_cloud_cover[T](
    height_levels: np.ndarray[T],
    cloud_cover: np.ndarray[T],
    z0: T,
    top_down: bool = False,
) -> T:
    """Compute cumulative cloud cover using the exponential-random overlap model.

    Args:
        height_levels (np.ndarray): Geometric height levels (m)
        cloud_cover (np.ndarray): Cloud cover fractions at each level (-)
        z0 (float): Decorrelation length (m)
        top_down (bool): If True, aggregate from the highest level downward.

    Returns:
        float: Cumulative cloud cover fraction over the traversed profile (-)
    """
    height_diff = np.diff(height_levels)

    assert np.all(height_diff > 0), (
        f"Expected height_levels to be strictly increasing "
        f"(surface→top), got min Δz={height_diff.min():.2f} m"
    )
    ordered_height_levels = height_levels[::-1] if top_down else height_levels
    ordered_cloud_cover = cloud_cover[::-1] if top_down else cloud_cover

    cloudy_idx = np.flatnonzero(ordered_cloud_cover > 0.0)

    if len(cloudy_idx) == 0:
        return 0.0

    ordered_height_levels = ordered_height_levels[cloudy_idx]
    ordered_cloud_cover = ordered_cloud_cover[cloudy_idx]

    cc_prev_levels: float = ordered_cloud_cover[0]
    for i in range(1, len(ordered_cloud_cover)):
        cc_prev_levels = exponential_random_cloud_overlap(
            ordered_cloud_cover[i],
            cc_prev_levels,
            ordered_height_levels[i],
            ordered_height_levels[i - 1],
            z0=z0,
        )
    return cc_prev_levels


def compute_cumulative_for_dataset(heights, cloud_cover, latitude):
    """Wrapper to compute cumulative cloud cover for xarray dataset."""
    z0 = get_cloud_layer_overlap_decorrelation_length(latitude)
    return compute_cumulative_cloud_cover(heights, cloud_cover, z0)


def compute_top_down_cumulative_for_dataset(heights, cloud_cover, latitude):
    """Wrapper to compute top-down cumulative cloud cover for xarray dataset."""
    z0 = get_cloud_layer_overlap_decorrelation_length(latitude)
    return compute_cumulative_cloud_cover(
        heights, cloud_cover, z0, top_down=True
    )


def create_era5_cloud_cover_datasets(
    ds: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset]:

    hybrid_diff = np.diff(ds["hybrid"].values)

    assert np.all(hybrid_diff < 0), (
        f"Expected hybrid to be strictly decreasing "
        f"(surface→top), got min Δz={hybrid_diff.min():.2f} m"
    )

    total_cloud_condensate_specific_mass = ds["clwc"] + ds["ciwc"]

    # Compute cloud liquid fraction (CLF), excluding non-cloud grid cells
    cloud_liquid_fraction = ds["clwc"] / (
        total_cloud_condensate_specific_mass.where(
            total_cloud_condensate_specific_mass > 0.0
        )
    )

    liquid_phase_cloud_flag = cloud_liquid_fraction > 0.9
    mixed_phase_cloud_flag = (
        cloud_liquid_fraction > 0.1
    ) & ~liquid_phase_cloud_flag
    ice_phase_cloud_flag = cloud_liquid_fraction <= 0.1

    liquid_cloud_cover = (
        ds["cc"].where(liquid_phase_cloud_flag).fillna(0.0)
    )

    # Hybrid should be sorted but I don't have any trust in anything whatsoever
    nearest_surface_hybrid_idx = (
        abs(ds["height"] - ds["surface_altitude"])
        .argmin(dim="hybrid")
        .compute()
    )

    # Fog criteria:
    # 1) Fog exists only if liquid cloud cover > 0.9 at nearest-to-surface model level.
    # 2) Fog extends upward from surface to the highest level where liquid cloud cover > 0.05.
    hybrid_index = xr.DataArray(
        np.arange(liquid_cloud_cover.sizes["hybrid"]),
        dims=["hybrid"],
    )
    at_or_above_surface = hybrid_index >= nearest_surface_hybrid_idx

    # Require fog to be continuous from the nearest-surface model level upward.
    # Any break (liquid cloud cover <= 0.05) terminates the fog layer aloft.
    fog_continuous_from_surface = (
        xr.where(
            at_or_above_surface,
            liquid_cloud_cover > 0.05,
            True,
        )
        .cumprod(dim="hybrid")
        .astype(bool)
        & at_or_above_surface
    )

    surface_liquid_cloud = liquid_cloud_cover.isel(
        hybrid=nearest_surface_hybrid_idx
    )
    fog_flag = (surface_liquid_cloud > 0.9) & fog_continuous_from_surface
    fog_cover = ds["cc"].where(fog_flag).fillna(0.0)

    mixed_cloud_cover = (
        ds["cc"].where(mixed_phase_cloud_flag).fillna(0.0)
    )
    liquid_and_mixed_cloud_cover = (
        liquid_cloud_cover + mixed_cloud_cover.fillna(0.0)
    )
    ice_cloud_cover = ds["cc"].where(ice_phase_cloud_flag).fillna(0.0)
    cloud_cover = ds["cc"].fillna(0.0)

    pressure = compute_model_level_pressure(ds)

    isccp_low_cloud_mask = pressure >= ISCCP_LOW_CLOUD_TOP_PRESSURE_THRESHOLD
    higher_than_isccp_low_cloud_mask = ~isccp_low_cloud_mask

    # These criteria come from IFS Documentation - Cy49r1. Part IV - Physical Processes. Chapter 7: Clouds and large-scale precipitation, Section 7.5.3 (b) Cloud height
    cloud_base_height = (
        ds["height"]
        .where(
            (cloud_cover > 0.01)
            & (total_cloud_condensate_specific_mass > 1e-6)
        )
        .min(dim="hybrid")
    )
    liquid_cloud_base_height = (
        ds["height"]
        .where(
            (liquid_cloud_cover > 0.0)
            & (total_cloud_condensate_specific_mass > 1e-6)
        )
        .min(dim="hybrid")
    )
    liquid_and_mixed_cloud_base_height = (
        ds["height"]
        .where(
            (liquid_and_mixed_cloud_cover > 0.01)
            & (total_cloud_condensate_specific_mass > 1e-6)
        )
        .min(dim="hybrid")
    )
    ice_cloud_base_height = (
        ds["height"]
        .where(
            (ice_cloud_cover > 0.01)
            & (total_cloud_condensate_specific_mass > 1e-6)
        )
        .min(dim="hybrid")
    )
    cloud_base_height_agl_da = cloud_base_height - ds["surface_altitude"]
    liquid_cloud_base_height_agl_da = (
        liquid_cloud_base_height - ds["surface_altitude"]
    )

    cloud_cover_ml = xr.Dataset(
        {
            "total_cloud_condensate_specific_mass": total_cloud_condensate_specific_mass,
            "cloud_liquid_fraction": cloud_liquid_fraction,
            "liquid_phase_cloud_flag": liquid_phase_cloud_flag,
            "mixed_phase_cloud_flag": mixed_phase_cloud_flag,
            "ice_phase_cloud_flag": ice_phase_cloud_flag,
            "fog_flag": fog_flag,
            "liquid_cloud_cover": liquid_cloud_cover,
            "liquid_and_mixed_cloud_cover": liquid_and_mixed_cloud_cover,
            "mixed_cloud_cover": mixed_cloud_cover,
            "ice_cloud_cover": ice_cloud_cover,
            "cloud_cover": cloud_cover,
            "pressure": pressure,
            "fog_cover": fog_cover,
        }
    )

    var_attrs_ml = {
        "liquid_cloud_cover": {
            "long_name": "Liquid phase fractional cloud cover",
            "units": "1",
            "comment": "Criteria for liquid phase cloud: cloud liquid fraction > 0.9",
        },
        "mixed_cloud_cover": {
            "long_name": "Mixed phase fractional cloud cover",
            "units": "1",
            "comment": "Criteria for mixed phase cloud: 0.1 <cloud liquid fraction <= 0.9",
        },
        "liquid_and_mixed_cloud_cover": {
            "long_name": "Sum of liquid and mixed phase fractional cloud covers",
            "units": "1",
        },
        "ice_cloud_cover": {
            "long_name": "Ice phase fractional cloud cover",
            "units": "1",
            "comment": "Criteria for ice phase cloud: cloud liquid fraction <= 0.1",
        },
        "cloud_liquid_fraction": {
            "long_name": "Cloud liquid fraction",
            "units": "1",
        },
        "pressure": {
            "long_name": "Pressure on ERA5 model full levels",
            "standard_name": "air_pressure",
            "units": "Pa",
            "positive": "down",
            "comment": "Computed from lnsp and ERA5 hybrid a(n), b(n) coefficients.",
        },
        "liquid_phase_cloud_flag": {
            "long_name": "Liquid phase cloud flag",
        },
        "mixed_phase_cloud_flag": {
            "long_name": "Mixed phase cloud flag",
        },
        "ice_phase_cloud_flag": {
            "long_name": "Ice phase cloud flag",
        },
        "fog_flag": {
            "long_name": "Fog cover flag",
            "units": "1",
            "comment": "1 when cc = 1.0 at the model level nearest to the surface, 0 otherwise.",
        },
    }
    for var, attrs in var_attrs_ml.items():
        cloud_cover_ml[var].attrs.update(attrs)
    cloud_cover_ml = cloud_cover_ml.assign_coords(height=ds["height"])

    cloud_cover_ml.attrs["title"] = (
        "ERA5 cloud liquid fraction and phase-resolved cloud cover on model levels for select ground observation sites"
    )
    cloud_cover_ml.attrs["description"] = (
        "Cloud liquid fraction, fractional cloud cover and cloud flags derived from ERA5 model level data for select ground cloud observation sites."
    )
    cloud_cover_ml.attrs["source"] = (
        "ERA5 model level reduced Gaussian grid (N320) data from ECMWF"
    )
    cloud_cover_ml.attrs["history"] = (
        f"Created by Sasu Karttunen at {datetime.now().isoformat()}"
    )

    cloud_cover_sl = xr.Dataset(
        {
            "cloud_base_height": cloud_base_height,
            "cloud_base_height_agl": cloud_base_height_agl_da,
            "liquid_cloud_base_height": liquid_cloud_base_height,
            "liquid_cloud_base_height_agl": liquid_cloud_base_height_agl_da,
            "liquid_and_mixed_cloud_base_height": liquid_and_mixed_cloud_base_height,
            "ice_cloud_base_height": ice_cloud_base_height,
            "fog_cover": fog_flag.any(dim="hybrid").astype(float),
        }
    ).squeeze()

    var_attrs_sl = {
        "cloud_base_height": {
            "long_name": "Cloud base height",
            "units": "m",
            "_FillValue": np.nan,
        },
        "cloud_base_height_agl": {
            "long_name": "Cloud base height above ground level",
            "units": "m",
            "_FillValue": np.nan,
        },
        "liquid_cloud_base_height": {
            "long_name": "Liquid phase cloud base height",
            "units": "m",
            "_FillValue": np.nan,
        },
        "liquid_cloud_base_height_agl": {
            "long_name": "Liquid phase cloud base height above ground level",
            "units": "m",
            "_FillValue": np.nan,
        },
        "liquid_and_mixed_cloud_base_height": {
            "long_name": "Liquid and mixed phase cloud base height",
            "units": "m",
            "_FillValue": np.nan,
        },
        "ice_cloud_base_height": {
            "long_name": "Ice phase cloud base height",
            "units": "m",
            "_FillValue": np.nan,
        },
        "fog_cover": {
            "long_name": "Fog cover",
            "units": "1",
            "comment": "1 when cloud_cover = 1.0 at the model level nearest to the surface, 0 otherwise.",
        },
    }
    for var, attrs in var_attrs_sl.items():
        cloud_cover_sl[var].attrs.update(attrs)

    cloud_cover_sl.attrs["title"] = (
        "ERA5 fractional height- and pressure-slab cloud cover and cloud base heights for select ground observation sites"
    )
    cloud_cover_sl.attrs["description"] = (
        "Phase-resolved fractional height- and pressure-slab cloud cover and cloud base heights from ERA5 model level data for select ground cloud observation sites."
    )
    cloud_cover_sl.attrs["source"] = (
        "ERA5 model level reduced Gaussian grid (N320) data from ECMWF"
    )
    cloud_cover_sl.attrs["history"] = (
        f"Created by Sasu Karttunen at {datetime.now().isoformat()}"
    )

    # Apply the function along the hybrid dimension for each time and site
    cloud_cover_sl["low_liquid_cloud_cover"] = xr.apply_ufunc(
        compute_cumulative_for_dataset,
        cloud_cover_ml["height"],
        cloud_cover_ml["liquid_cloud_cover"]
        .where((cloud_cover_ml["height"] <= (2000.0 + ds["surface_altitude"])))
        .fillna(0.0),
        cloud_cover_ml["latitude"],
        input_core_dims=[["hybrid"], ["hybrid"], []],
        vectorize=True,
        dask="parallelized",
    )
    # Exclude fog from the cover
    cloud_cover_sl["low_liquid_cloud_cover"] = cloud_cover_sl[
        "low_liquid_cloud_cover"
    ]  # .where(~fog_cover, other=np.nan)

    cloud_cover_sl["low_liquid_and_mixed_cloud_cover"] = xr.apply_ufunc(
        compute_cumulative_for_dataset,
        cloud_cover_ml["height"],
        cloud_cover_ml["liquid_and_mixed_cloud_cover"]
        .where((cloud_cover_ml["height"] <= (2000.0 + ds["surface_altitude"])))
        .fillna(0.0),
        cloud_cover_ml["latitude"],
        input_core_dims=[["hybrid"], ["hybrid"], []],
        vectorize=True,
        dask="parallelized",
    )
    cloud_cover_sl["low_liquid_and_mixed_cloud_cover"] = cloud_cover_sl[
        "low_liquid_and_mixed_cloud_cover"
    ]

    cloud_cover_sl["low_ice_cloud_cover"] = xr.apply_ufunc(
        compute_cumulative_for_dataset,
        cloud_cover_ml["height"],
        cloud_cover_ml["ice_cloud_cover"]
        .where((cloud_cover_ml["height"] <= (2000.0 + ds["surface_altitude"])))
        .fillna(0.0),
        cloud_cover_ml["latitude"],
        input_core_dims=[["hybrid"], ["hybrid"], []],
        vectorize=True,
        dask="parallelized",
    )
    cloud_cover_sl["low_cloud_cover"] = xr.apply_ufunc(
        compute_cumulative_for_dataset,
        cloud_cover_ml["height"],
        cloud_cover_ml["cloud_cover"]
        .where((cloud_cover_ml["height"] <= (2000.0 + ds["surface_altitude"])))
        .fillna(0.0),
        cloud_cover_ml["latitude"],
        input_core_dims=[["hybrid"], ["hybrid"], []],
        vectorize=True,
        dask="parallelized",
    )

    cloud_cover_sl["low_cloud_cover_isccp"] = xr.apply_ufunc(
        compute_top_down_cumulative_for_dataset,
        cloud_cover_ml["height"],
        cloud_cover_ml["cloud_cover"].where(isccp_low_cloud_mask),
        cloud_cover_ml["latitude"],
        input_core_dims=[["hybrid"], ["hybrid"], []],
        vectorize=True,
        dask="parallelized",
    )

    cloud_cover_above_low_cloud_slab = xr.apply_ufunc(
        compute_top_down_cumulative_for_dataset,
        cloud_cover_ml["height"],
        cloud_cover_ml["cloud_cover"]
        .where(higher_than_isccp_low_cloud_mask)
        .fillna(0.0),
        cloud_cover_ml["latitude"],
        input_core_dims=[["hybrid"], ["hybrid"], []],
        vectorize=True,
        dask="parallelized",
    )

    total_cloud_cover_top_down = xr.apply_ufunc(
        compute_top_down_cumulative_for_dataset,
        cloud_cover_ml["height"],
        cloud_cover_ml["cloud_cover"],
        cloud_cover_ml["latitude"],
        input_core_dims=[["hybrid"], ["hybrid"], []],
        vectorize=True,
        dask="parallelized",
    )

    cloud_cover_sl["low_cloud_cover_isccp_non_obscured"] = (
        (total_cloud_cover_top_down - cloud_cover_above_low_cloud_slab)
        / (1.0 - cloud_cover_above_low_cloud_slab)
    ).clip(min=0.0, max=1.0)

    # Add attributes
    cloud_cover_sl["low_liquid_cloud_cover"].attrs = {
        "long_name": "Cumulative liquid phase cloud cover",
        "units": "1",
        "comment": "Computed using exponential-random cloud overlap model",
    }
    cloud_cover_sl["low_liquid_and_mixed_cloud_cover"].attrs = {
        "long_name": "Cumulative liquid and mixed phase cloud cover",
        "units": "1",
        "comment": "Computed using exponential-random cloud overlap model",
    }
    cloud_cover_sl["low_ice_cloud_cover"].attrs = {
        "long_name": "Cumulative ice phase cloud cover",
        "units": "1",
        "comment": "Computed using exponential-random cloud overlap model",
    }
    cloud_cover_sl["low_cloud_cover"].attrs = {
        "long_name": "Cumulative total cloud cover",
        "units": "1",
        "comment": "Computed using exponential-random cloud overlap model",
    }
    cloud_cover_sl["low_cloud_cover_isccp"].attrs = {
        "long_name": "ISCCP-definition cumulative low cloud cover",
        "units": "1",
        "comment": "Computed from all-phase cloud cover on model levels with pressure >= 680 hPa using top-down exponential-random cloud overlap aggregation.",
    }
    cloud_cover_sl["low_cloud_cover_isccp_non_obscured"].attrs = {
        "long_name": "ISCCP-definition non-obscured cumulative low cloud cover",
        "units": "1",
        "comment": "Computed as the additional projected cloud cover contributed by model levels with pressure >= 680 hPa when the full atmospheric column is aggregated top-down using the exponential-random cloud overlap model.",
    }
    for var in cloud_cover_sl.data_vars:
        if cloud_cover_sl[var].dtype in [np.float64, float]:
            cloud_cover_sl[var] = cloud_cover_sl[var].astype(np.float32)
    for var in cloud_cover_ml.data_vars:
        if cloud_cover_ml[var].dtype in [np.float64, float]:
            cloud_cover_ml[var] = cloud_cover_ml[var].astype(np.float32)

    return cloud_cover_ml, cloud_cover_sl


def run():
    cluster = LocalCluster(
        n_workers=2, threads_per_worker=3, memory_limit="auto"
    )
    Client(cluster)

    ds = xr.open_mfdataset(
        ERA5_DATA_PATH.glob("era5_*.nc"), decode_timedelta=True
    )

    ds["surface_altitude"] = ds["z"] / GRAVITATIONAL_ACCELERATION

    cloud_cover_ml, cloud_cover_sl = create_era5_cloud_cover_datasets(ds)

    cloud_cover_sl.to_netcdf(
        DATA_PATHS.era5_level2_clouds_hourly,
        mode="w",
        format="NETCDF4",
        engine="netcdf4",
    )


if __name__ == "__main__":
    setup_logging(logging.INFO)

    run()
