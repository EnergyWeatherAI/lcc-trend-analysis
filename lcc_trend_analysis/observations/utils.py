"""Utility functions for ground site metadata and instrument preferences.

Ground site metadata (ground_sites.parquet):
- Site locations and identifiers
- ERA5 grid cell associations (for model comparison)
- Buffer zones (for satellite overpass matching)
- Preferred ceilometer types (for multi-instrument sites)

Geometries stored as WKB (Well-Known Binary) for efficient serialization.
"""
from collections.abc import Iterable, Iterator
import json
import importlib
from binascii import unhexlify
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from shapely import from_wkb

from ..logging import get_logger
from ..paths import get_data_paths
from ..type_aliases import GeoDataFrame

import geopandas as gpd

logger = get_logger(__name__)

# Ceilometer preference order based on range capability, temporal resolution, and noise characteristics
PREFERED_CEILOMETER_ORDER = [
    "pollyxt",          # Research lidar, best sensitivity, 1064 nm
    "cl61",             # Latest Vaisala, 15 km range
    "chm15kx",          # Lufft, good sensitivity
    "chm15k",           # Lufft, 1064 nm
    "cs135",            # Campbell Sci, compact but capable
    "cl51",             # Vaisala, standard operational
    "cl31",             # Vaisala, very common
    "ct75k",            # Vaisala, extended range
    "ct25k",            # Vaisala, basic operational
    "streamline",       # Wind lidar (lower priority)
    "windcubewls200s",  # Wind lidar
    "windcubewls70",    # Wind lidar
]

CLOUDSAT_SITE_EVALUATION_POLYGON_COLUMN = "site_near_range"

_SPURIOUS_DAY_LOOKUP_CACHE: (
    tuple[
        tuple[tuple[str, int | None, int | None], ...],
        dict[str, dict[str, frozenset[pd.Timestamp]]],
    ]
    | None
) = None

class DimensionalityError(ValueError):
    pass

class FileContentError(ValueError):
    pass

class DimensionalityWarning(Warning):
    pass

def iter_ceilometers() -> Iterator[str]:
    """Returns iterator over preferred ceilometer order."""
    return iter(PREFERED_CEILOMETER_ORDER)

def _deserialize_wkb_geometry_column(
    ground_sites: gpd.GeoDataFrame,
    column_name: str,
) -> None:
    if column_name not in ground_sites.columns:
        return

    def _deserialize_geometry(geometry):
        if geometry is None:
            return None
        if hasattr(geometry, "geom_type"):
            return geometry
        if isinstance(geometry, bytes):
            return from_wkb(geometry)
        return from_wkb(unhexlify(str(geometry)))

    deserialized_geometries = [
        _deserialize_geometry(geometry)
        for geometry in ground_sites[column_name].tolist()
    ]
    ground_sites[column_name] = pd.Series(
        deserialized_geometries,
        index=ground_sites.index,
        dtype=object,
    )  # type: ignore[assignment]


def get_ground_sites_gdf(
    require_columns: Iterable[str] | None = None,
) -> GeoDataFrame:
    """Load ground site metadata from parquet file.
    
    Returns GeoDataFrame with:
    - geometry: Site location (Point)
    - era5_cell_center: ERA5 grid cell center (Point)
    - era5_cell_polygon: ERA5 grid cell boundary (Polygon)
    - site_near_range: 2.5 km buffer for satellite matching (Polygon)
    
    Geometries stored as WKB hex strings and deserialized to shapely objects.
    
    Returns:
        GeoDataFrame: Site metadata with deserialized geometries
    """
    ground_sites: gpd.GeoDataFrame = gpd.read_parquet(
        get_data_paths().ground_sites
    )

    for column_name in [
        "era5_cell_center",
        "era5_cell_polygon",
        "cloudsat_cpr_footprint_buffer",
        CLOUDSAT_SITE_EVALUATION_POLYGON_COLUMN,
    ]:
        _deserialize_wkb_geometry_column(ground_sites, column_name)

    if require_columns is not None:
        missing_columns = [
            column_name
            for column_name in require_columns
            if column_name not in ground_sites.columns
        ]
        if missing_columns:
            missing_columns_str = ", ".join(sorted(missing_columns))
            raise KeyError(
                "Missing required ground-site metadata columns: "
                f"{missing_columns_str}"
            )

    return ground_sites


def clear_spurious_day_lookup_cache() -> None:
    """Clear the cached spurious-day configuration."""
    global _SPURIOUS_DAY_LOOKUP_CACHE
    _SPURIOUS_DAY_LOOKUP_CACHE = None


def _spurious_lookup_signature(paths: Iterable[Path]) -> tuple[tuple[str, int | None, int | None], ...]:
    """Build a stable signature for the current default spurious-data files."""
    signature: list[tuple[str, int | None, int | None]] = []
    for path in paths:
        if path.exists():
            stat_result = path.stat()
            signature.append((str(path), stat_result.st_mtime_ns, stat_result.st_size))
        else:
            signature.append((str(path), None, None))
    return tuple(signature)

def _expand_spurious_day_entries(entries: list[Any]) -> frozenset[pd.Timestamp]:
    excluded_days: set[pd.Timestamp] = set()
    for entry in entries:
        if isinstance(entry, str):
            excluded_days.add(pd.Timestamp(entry).normalize())
            continue

        if isinstance(entry, dict):
            if "start" not in entry or "end" not in entry:
                raise ValueError(
                    "Spurious-data range entries must contain both 'start' and 'end'"
                )

            start = pd.Timestamp(entry["start"]).normalize()
            end = pd.Timestamp(entry["end"]).normalize()
            if end < start:
                raise ValueError(
                    "Spurious-data range entry has end before start: "
                    f"{entry!r}"
                )

            for day in pd.date_range(start=start, end=end, freq="D"):
                excluded_days.add(day.normalize())
            continue

        raise ValueError(
            "Unsupported spurious-data entry; expected date string or start/end mapping, "
            f"got {type(entry).__name__}"
        )

    return frozenset(excluded_days)


def _normalize_spurious_day_lookup(
    raw_lookup: dict[str, Any],
) -> dict[str, dict[str, frozenset[pd.Timestamp]]]:
    normalized_lookup: dict[str, dict[str, frozenset[pd.Timestamp]]] = {}

    for site_id, site_entries in raw_lookup.items():
        if not isinstance(site_entries, dict):
            raise ValueError(
                "Spurious-data site entries must map instrument names to day lists, "
                f"got {type(site_entries).__name__} for site {site_id!r}"
            )

        site_key = str(site_id).lower()
        normalized_lookup[site_key] = {}

        for instrument_name, entries in site_entries.items():
            if not isinstance(entries, list):
                raise ValueError(
                    "Spurious-data instrument entries must be lists, "
                    f"got {type(entries).__name__} for {site_id!r}/{instrument_name!r}"
                )

            instrument_key = str(instrument_name).lower()
            normalized_lookup[site_key][instrument_key] = _expand_spurious_day_entries(
                entries
            )

    return normalized_lookup


def _load_single_spurious_day_lookup(
    config_path: Path,
) -> dict[str, dict[str, frozenset[pd.Timestamp]]]:
    """Load and normalize one spurious-day JSON file."""
    if not config_path.exists():
        logger.debug(
            "Spurious-data file not found at %s; skipping",
            config_path,
        )
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        raw_lookup = json.load(handle)

    if raw_lookup is None:
        raw_lookup = {}
    if not isinstance(raw_lookup, dict):
        raise ValueError(
            "Spurious-data configuration must be a JSON object mapping sites to instruments"
        )

    return _normalize_spurious_day_lookup(raw_lookup)


def _merge_spurious_day_lookups(
    *lookups: dict[str, dict[str, frozenset[pd.Timestamp]]],
) -> dict[str, dict[str, frozenset[pd.Timestamp]]]:
    """Merge multiple spurious-day lookups by taking the union of excluded days."""
    merged: dict[str, dict[str, set[pd.Timestamp]]] = {}
    for lookup in lookups:
        for site_key, instruments in lookup.items():
            site_merged = merged.setdefault(site_key, {})
            for instrument_key, days in instruments.items():
                if instrument_key in site_merged:
                    site_merged[instrument_key].update(days)
                else:
                    site_merged[instrument_key] = set(days)

    return {
        site_key: {
            instrument_key: frozenset(days)
            for instrument_key, days in instruments.items()
        }
        for site_key, instruments in merged.items()
    }


def load_spurious_day_lookup(
    spurious_data_path: Path | None = None,
) -> dict[str, dict[str, frozenset[pd.Timestamp]]]:
    """Load and normalize the spurious-day configuration once per process.

    Reads both spurious_data_ranges.json (date ranges) and
    spurious_data_dates.json (individual dates) and returns their union.
    A single explicit path can be supplied to override both defaults, which is
    intended for tests only.
    """
    global _SPURIOUS_DAY_LOOKUP_CACHE
    if spurious_data_path is None:
        data_paths = get_data_paths()
        current_signature = _spurious_lookup_signature(
            (data_paths.spurious_data_ranges, data_paths.spurious_data_dates)
        )
        if _SPURIOUS_DAY_LOOKUP_CACHE is not None:
            cached_signature, cached_lookup = _SPURIOUS_DAY_LOOKUP_CACHE
            if cached_signature == current_signature:
                return cached_lookup

    if spurious_data_path is not None:
        normalized_lookup = _load_single_spurious_day_lookup(spurious_data_path)
    else:
        ranges_lookup = _load_single_spurious_day_lookup(data_paths.spurious_data_ranges)
        dates_lookup = _load_single_spurious_day_lookup(data_paths.spurious_data_dates)
        normalized_lookup = _merge_spurious_day_lookups(ranges_lookup, dates_lookup)

        if not ranges_lookup and not dates_lookup:
            logger.warning(
                "Neither %s nor %s were found; no spurious day filtering will be applied",
                data_paths.spurious_data_ranges,
                data_paths.spurious_data_dates,
            )

    n_sites = len(normalized_lookup)
    n_site_instrument_pairs = sum(
        len(site_entries)
        for site_entries in normalized_lookup.values()
    )
    n_excluded_days = sum(
        len(excluded_days)
        for site_entries in normalized_lookup.values()
        for excluded_days in site_entries.values()
    )
    if n_sites:
        logger.info(
            "Loaded spurious-day configuration (%d sites, %d site-instrument pairs, %d excluded days total)",
            n_sites,
            n_site_instrument_pairs,
            n_excluded_days,
        )

    if spurious_data_path is None:
        _SPURIOUS_DAY_LOOKUP_CACHE = (current_signature, normalized_lookup)
    return normalized_lookup


def get_spurious_days_for(
    site_id: str,
    instrument_name: str,
) -> frozenset[pd.Timestamp]:
    """Return configured spurious calendar days for one site and instrument."""
    spurious_day_lookup = load_spurious_day_lookup()
    return spurious_day_lookup.get(site_id.lower(), {}).get(
        instrument_name.lower(),
        frozenset(),
    )


def is_spurious_day(
    site_id: str,
    instrument_name: str,
    date: pd.Timestamp,
) -> bool:
    """Return whether a timestamp falls on a configured spurious calendar day."""
    return pd.Timestamp(ts_input=date).normalize() in get_spurious_days_for(
        site_id,
        instrument_name,
    )


def mask_spurious_days(
    dataset: xr.Dataset,
    *,
    site_id: str,
    instrument_name: str,
) -> xr.Dataset:
    """Mask all timestamps whose calendar day is configured as spurious."""
    excluded_days = get_spurious_days_for(site_id, instrument_name)
    if "time" not in dataset.coords or not excluded_days:
        return dataset

    dataset_attrs = dict(dataset.attrs)
    time_days = pd.DatetimeIndex(pd.to_datetime(np.asarray(dataset["time"].values))).normalize()
    keep_mask = xr.DataArray(
        ~time_days.isin(list(excluded_days)),
        coords={"time": dataset["time"]},
        dims=["time"],
    )
    masked_dataset = dataset.isel(time=np.flatnonzero(np.asarray(keep_mask.values)))
    masked_dataset.attrs = dataset_attrs
    return masked_dataset


def extract_site_lat_lon(geometry) -> tuple[float, float]:
    """Return latitude and longitude from a site point geometry.

    Using bounds keeps the extraction compatible with the broad geometry type
    annotations carried by geopandas and shapely.
    """
    min_x, min_y, _, _ = geometry.bounds
    return float(min_y), float(min_x)


def get_site_coordinate_lookup(
    ground_sites: GeoDataFrame,
    site_ids: Iterable[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Build an ordered site_id -> (latitude, longitude) lookup."""
    if site_ids is None:
        return {
            str(site_id): extract_site_lat_lon(geometry)
            for site_id, geometry in ground_sites.geometry.items()
            if geometry is not None
        }

    coordinates: dict[str, tuple[float, float]] = {}
    missing_site_ids: list[str] = []
    for site_id in site_ids:
        site_key = str(site_id)
        if site_key not in ground_sites.index:
            missing_site_ids.append(site_key)
            continue

        geometry = ground_sites.loc[site_key, "geometry"]
        if geometry is None:
            missing_site_ids.append(site_key)
            continue

        coordinates[site_key] = extract_site_lat_lon(geometry)

    if missing_site_ids:
        missing_sites = ", ".join(sorted(set(missing_site_ids)))
        raise KeyError(f"Missing ground-site coordinates for: {missing_sites}")

    return coordinates


def compute_solar_zenith_angle_for_site(
    times,
    latitude: float,
    longitude: float,
) -> xr.DataArray:
    """Compute solar zenith angle for one site and a sequence of timestamps.

    Returns a ``time``-indexed DataArray so the result can be aligned directly
    against profile-wise ceilometer data.
    """
    pvlib = importlib.import_module("pvlib")
    time_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    solar_position = pvlib.solarposition.get_solarposition(
        time=time_index,
        latitude=float(latitude),
        longitude=float(longitude),
    )

    return xr.DataArray(
        solar_position["zenith"].to_numpy(),
        coords={"time": time_index},
        dims=["time"],
        name="solar_zenith_angle",
    )


def compute_solar_zenith_angle_for_sites(
    times,
    site_coordinates: dict[str, tuple[float, float]],
) -> xr.DataArray:
    """Compute solar zenith angle for multiple sites sharing one time axis."""
    if not site_coordinates:
        raise ValueError("site_coordinates must contain at least one site")

    pvlib = importlib.import_module("pvlib")
    time_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    site_ids = list(site_coordinates)
    latitudes = np.asarray(
        [site_coordinates[site_id][0] for site_id in site_ids],
        dtype=float,
    )
    longitudes = np.asarray(
        [site_coordinates[site_id][1] for site_id in site_ids],
        dtype=float,
    )

    lat2d = np.broadcast_to(latitudes[np.newaxis, :], (len(time_index), len(site_ids)))
    lon2d = np.broadcast_to(longitudes[np.newaxis, :], (len(time_index), len(site_ids)))
    time2d = np.broadcast_to(time_index.to_numpy()[:, None], (len(time_index), len(site_ids)))

    solar_position = pvlib.solarposition.get_solarposition(
        time=pd.to_datetime(time2d.ravel()),
        latitude=lat2d.ravel(),
        longitude=lon2d.ravel(),
    )
    solar_zenith_angle = solar_position["zenith"].to_numpy().reshape(
        len(time_index),
        len(site_ids),
    )

    return xr.DataArray(
        solar_zenith_angle,
        coords={"time": time_index, "site": site_ids},
        dims=["time", "site"],
        name="solar_zenith_angle",
    )