import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as NetCDF4Dataset

from lcc_trend_analysis.type_aliases import Dataset, GeoDataFrame, Timestamp

# Groups in the CERES SYN1deg-Day HDF product that contain the cloud-layer variables we
# want to retain. Kept for documentation purposes; variable discovery below walks the
# whole file (root + any nested groups) so it also works if the underlying HDF4 reader
# exposes these as a flat namespace instead of nested netCDF4 groups.
CERES_SYN1DEG_GROUPS = (
    "Observed_Cloud_Layer_Properties",
    "Adjusted_Input_Meteorological_Variables",
)
CERES_VARIABLE_PREFIX_PATTERN = re.compile(r"^(obs_cld_|adj_cld_)")
CERES_CLOUD_LAYER_LABELS = {
    1: "high",
    2: "upper_mid",
    3: "lower_mid",
    4: "low",
    5: "total",
}
# Cloud amount variables are reported by CERES as a percentage (0-100); we convert them
# to a fraction (0-1) and rename them to *_cloud_cover to match the rest of the codebase.
CERES_CLOUD_COVER_SOURCE_NAMES = frozenset({"obs_cld_amount", "adj_cld_amount"})
CERES_SYN1DEG_DAY_PRODUCT = "SYN1deg-Day"
CERES_SYN1DEG_1HOUR_PRODUCT = "SYN1deg-1Hour"
CERES_SUPPORTED_PRODUCTS = frozenset(
    {CERES_SYN1DEG_DAY_PRODUCT, CERES_SYN1DEG_1HOUR_PRODUCT}
)
CERES_GMT_HOUR_INDEX = "gmt_hr_index"
CERES_SYN1DEG_FILENAME_RE = re.compile(
    r"^CER_(?P<product>[^_]+)_(?P<platforms>[^_]+)_(?P<edition>Edition[^_]+)_"
    r"(?P<granule_id>\d+)\.(?P<date>\d{8})\.hdf$"
)


@dataclass(frozen=True)
class CeresSyn1DegFilenameInfo:
    product: str
    platforms: str
    edition: str
    granule_id: str
    time: Timestamp


def is_ceres_hourly_product(product: str) -> bool:
    return "SYN1deg-1Hour" == CERES_SYN1DEG_1HOUR_PRODUCT

def haversine_distance_m(
    latitude_1: np.ndarray | float,
    longitude_1: np.ndarray | float,
    latitude_2: float,
    longitude_2: float,
) -> np.ndarray:
    earth_radius_m = 6_371_000.0
    lat1_rad = np.deg2rad(latitude_1)
    lon1_rad = np.deg2rad(longitude_1)
    lat2_rad = np.deg2rad(latitude_2)
    lon2_rad = np.deg2rad(longitude_2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    hav = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * np.arcsin(np.sqrt(hav))

def parse_ceres_syn1deg_filename(
    file_path: Path,
) -> CeresSyn1DegFilenameInfo | None:
    """Parse a CERES SYN1deg filename into its product and source-date fields."""
    match = CERES_SYN1DEG_FILENAME_RE.match(file_path.name)
    if match is None:
        return None
    return CeresSyn1DegFilenameInfo(
        product=match.group("product"),
        platforms=match.group("platforms"),
        edition=match.group("edition"),
        granule_id=match.group("granule_id"),
        time=pd.to_datetime(match.group("date"), format="%Y%m%d"),
    )

def _collect_matching_variables(ds_root: NetCDF4Dataset) -> dict[str, Any]:
    """Walk the root group and any nested groups, collecting variables whose name
    matches the obs_cld_*/adj_cld_* prefix pattern. Works whether the underlying HDF4
    reader exposes CERES_SYN1DEG_GROUPS as nested netCDF4 groups or flattens everything
    into the root namespace.
    """
    variables: dict[str, Any] = {}

    def _walk(group: Any) -> None:
        for name, variable in group.variables.items():
            if name not in variables and CERES_VARIABLE_PREFIX_PATTERN.match(name):
                variables[name] = variable
        for sub_group in group.groups.values():
            _walk(sub_group)

    _walk(ds_root)
    return variables


def _to_native_byteorder(values: np.ndarray) -> np.ndarray:
    """Return a copy of values with native byte order.

    CERES SYN1deg HDF4 files store data big-endian; pandas/xarray indexing
    (used for nearest-neighbor site selection) does not support non-native
    byte order, so coordinate and data arrays must be normalized after read.
    """
    if values.dtype.byteorder not in ("=", "|"):
        return values.astype(values.dtype.newbyteorder("="))
    return values


def _read_root_coordinate(ds_root: NetCDF4Dataset, variable_name: str) -> xr.Variable:
    variable = ds_root.variables[variable_name]
    attrs = {attr_name: variable.getncattr(attr_name) for attr_name in variable.ncattrs()}
    values = _to_native_byteorder(np.asarray(variable[:]))
    return xr.Variable(variable.dimensions, values, attrs=attrs)


def _rename_ceres_variable(source_name: str) -> str:
    """Rename a source CERES variable to its level1 output name.

    Replaces "cld" with "cloud" throughout, and additionally renames the cloud
    amount variables (obs_cld_amount/adj_cld_amount) to *_cloud_cover, since they
    are converted from a percentage to a fraction and represent cloud cover.
    """
    renamed = source_name.replace("cld", "cloud")
    if source_name in CERES_CLOUD_COVER_SOURCE_NAMES:
        renamed = renamed.replace("amount", "cover")
    return renamed


def _decode_ceres_variable(variable: Any, *, is_cloud_cover: bool = False) -> xr.Variable:
    variable.set_auto_mask(False)
    variable.set_auto_scale(False)
    attrs = {attr_name: variable.getncattr(attr_name) for attr_name in variable.ncattrs()}

    raw_values = np.asarray(variable[:], dtype=np.float32)
    fill_value = attrs.get("_FillValue")
    if fill_value is not None:
        raw_values = np.where(raw_values == np.float32(fill_value), np.nan, raw_values)
    attrs["_FillValue"] = np.float32(np.nan)

    if is_cloud_cover:
        raw_values = raw_values / 100.0
        attrs["units"] = "1"
        if "valid_range" in attrs:
            attrs["valid_range"] = np.asarray(attrs["valid_range"], dtype=np.float32) / 100.0
        long_name = attrs.get("long_name")
        if long_name:
            attrs["long_name"] = str(long_name).replace("Amount", "Cover")

    return xr.Variable(variable.dimensions, raw_values, attrs=attrs)


def assign_ceres_hourly_time(
    ds: Dataset,
    source_date: Timestamp,
) -> Dataset:
    """Replace CERES hourly GMT indices with UTC midpoint timestamps.

    A SYN1deg-1Hour file represents the UTC hour indexed by
    ``gmt_hr_index`` at the midpoint of that period. For example, index zero
    on 2026-05-31 becomes 2026-05-31 00:30 UTC.
    """
    if CERES_GMT_HOUR_INDEX not in ds.coords:
        raise ValueError(
            f"CERES hourly dataset is missing root coordinate {CERES_GMT_HOUR_INDEX!r}"
        )

    hour_index = ds[CERES_GMT_HOUR_INDEX] - 1
    if hour_index.dims != (CERES_GMT_HOUR_INDEX,):
        raise ValueError(
            f"Expected {CERES_GMT_HOUR_INDEX!r} to be one-dimensional on its "
            f"own dimension, got {hour_index.dims!r}"
        )

    hours = np.asarray(hour_index.to_numpy())
    if not np.issubdtype(hours.dtype, np.integer):
        raise ValueError(
            f"Expected integer {CERES_GMT_HOUR_INDEX!r} values, got {hours.dtype}"
        )

    source_midnight = pd.Timestamp(source_date).normalize()
    timestamps = (
        source_midnight
        + pd.to_timedelta(hours.astype(np.int64), unit="h")
        + pd.Timedelta(minutes=30)
    ).to_numpy(dtype="datetime64[ns]")

    ds = ds.assign_coords(time=(CERES_GMT_HOUR_INDEX, timestamps))
    ds = ds.swap_dims({CERES_GMT_HOUR_INDEX: "time"})
    return ds.drop_vars(CERES_GMT_HOUR_INDEX)


def read_ceres_syn1deg_level0(
    file_path: Path,
    *,
    product: str = CERES_SYN1DEG_DAY_PRODUCT,
    source_date: Timestamp | None = None,
) -> Dataset:
    """Read a CERES SYN1deg level0 HDF file and retain cloud-layer variables.

    SYN1deg-1Hour inputs additionally convert ``gmt_hr_index`` into a full
    UTC ``time`` coordinate using ``source_date``.
    """
    with NetCDF4Dataset(file_path) as ds_root:
        latitude = _read_root_coordinate(ds_root, "latitude")
        longitude = _read_root_coordinate(ds_root, "longitude")
        cloud_layer = _read_root_coordinate(ds_root, "cloud_layer")
        gmt_hr_index = (
            _read_root_coordinate(ds_root, CERES_GMT_HOUR_INDEX)
            if CERES_GMT_HOUR_INDEX in ds_root.variables
            else None
        )

        matched_variables = _collect_matching_variables(ds_root)
        if not matched_variables:
            raise ValueError(
                f"No obs_cld_*/adj_cld_* variables found in {file_path}"
            )

        data_vars = {
            _rename_ceres_variable(var_name): _decode_ceres_variable(
                variable,
                is_cloud_cover=var_name in CERES_CLOUD_COVER_SOURCE_NAMES,
            )
            for var_name, variable in matched_variables.items()
        }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": latitude,
            "longitude": longitude,
            "cloud_layer": cloud_layer,
            **(
                {CERES_GMT_HOUR_INDEX: gmt_hr_index}
                if gmt_hr_index is not None
                else {}
            ),
        },
    )
    cloud_layer_names = np.array(
        [CERES_CLOUD_LAYER_LABELS.get(int(v), "unknown") for v in ds["cloud_layer"].to_numpy()],
        dtype=object,
    )
    ds = ds.assign_coords(cloud_layer_name=("cloud_layer", cloud_layer_names))
    if is_ceres_hourly_product(product):
        if source_date is None:
            raise ValueError("source_date is required for CERES SYN1deg-1Hour inputs")
        ds = assign_ceres_hourly_time(ds, source_date)
    return ds


def extract_sites_from_ceres_grid(ds: Dataset, ground_sites: GeoDataFrame) -> Dataset:
    """Extract data at the nearest CERES SYN1deg grid cell center for each ground site.

    Returns a Dataset with dimensions (site, cloud_layer). The CERES SYN1deg grid uses
    0-360 degrees_east longitude, so site longitudes are normalized before matching.
    """
    site_ids = [str(site_id) for site_id in ground_sites.index]
    site_latitude = np.array(
        [float(ground_sites.loc[site_id].geometry.y) for site_id in site_ids]
    )
    site_longitude = np.array(
        [float(ground_sites.loc[site_id].geometry.x) for site_id in site_ids]
    )
    site_longitude_0_360 = np.mod(site_longitude, 360.0)

    site_lat_da = xr.DataArray(site_latitude, dims="site", coords={"site": site_ids})
    site_lon_da = xr.DataArray(
        site_longitude_0_360, dims="site", coords={"site": site_ids}
    )

    extracted = ds.sel(latitude=site_lat_da, longitude=site_lon_da, method="nearest")

    grid_cell_latitude = extracted["latitude"].to_numpy()
    grid_cell_longitude = extracted["longitude"].to_numpy()
    site_match_distance_km = (
        haversine_distance_m(
            grid_cell_latitude,
            grid_cell_longitude,
            site_latitude,
            site_longitude_0_360,
        )
        / 1000.0
    )

    extracted = extracted.rename(
        {"latitude": "grid_cell_latitude", "longitude": "grid_cell_longitude"}
    )
    extracted = extracted.assign_coords(
        site_latitude=("site", site_latitude),
        site_longitude=("site", site_longitude),
        site_match_distance_km=("site", site_match_distance_km),
    )

    ordered_dims = ["site", "cloud_layer"] + [
        dim_name for dim_name in extracted.dims if dim_name not in {"site", "cloud_layer"}
    ]
    return extracted.transpose(*ordered_dims, missing_dims="ignore")
