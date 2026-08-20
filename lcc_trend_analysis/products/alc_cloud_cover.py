import datetime
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import numpy as np
import pandas as pd
import xarray as xr

from lcc_trend_analysis.metadata import set_variable_metadata
from lcc_trend_analysis.type_aliases import (
    DataArray,
    Dataset,
)

from ..algorithms.cloud_detection import ALCCloudCoverEstimator
from ..observations.ceilometers import CeilometerDataFile


@dataclass
class CloudCoverTimeseriesGranule:
    """Container for one daily cloud-product time-series slice."""

    site_id: str
    date: pd.Timestamp
    instrument_name: str
    data: xr.Dataset


def extract_raw_time_series_data(data: Dataset) -> Dataset:
    """Return all native-sample variables defined only on the time axis."""
    if "time" not in data.coords:
        raise ValueError(
            "Expected a 'time' coordinate in the cloud cover product."
        )

    variable_names = [
        var_name
        for var_name, da in data.data_vars.items()
        if da.dims == ("time",)
    ]
    if not variable_names:
        raise ValueError(
            "No 1D time-series variables found in the cloud cover product. "
            "This usually indicates that the time dimension was collapsed unexpectedly."
        )

    raw_time_series = data[variable_names].copy(deep=False)
    raw_time_series = raw_time_series.assign_coords(time=data["time"])
    raw_time_series.attrs = dict(data.attrs)
    return raw_time_series


class ALCCloudCoverProduct:
    def __init__(self, estimator=ALCCloudCoverEstimator()):
        self._data: Optional[Dataset] = None
        self.estimator: ALCCloudCoverEstimator = estimator

    @classmethod
    def from_data_file(
        cls,
        data_file: CeilometerDataFile,
        estimator: Optional[ALCCloudCoverEstimator] = None,
    ) -> Self:
        product: Self = cls(estimator or ALCCloudCoverEstimator())

        X = data_file.data
        assert X is not None

        with warnings.catch_warnings(record=True) as emitted_warnings:
            warnings.simplefilter("always")
            cloud_layers: DataArray = product.estimator.transform(X)

        if emitted_warnings:
            for warn in emitted_warnings:
                warnings.warn(
                    f"While estimating cloud layers for {data_file.data_path}, the following warning was issued:\n"
                    f"  {warn.message} ({warn.filename}:{warn.lineno})",
                    category=warn.category,
                )

        cloud_mask: DataArray = cloud_layers > 0

        argmax_index = cloud_mask.argmax(dim="height")

        is_valid = cloud_mask.any(dim="height")

        # Handle cloud-free profiles: use dummy index 0 for indexing, then mask invalid results
        # This avoids index errors while allowing vectorized operations
        safe_index = argmax_index.where(is_valid, other=0).astype("int32")  # type: ignore

        # Index safely (this includes dummy results)
        cloud_base_height_asl = cloud_mask["height"].isel(height=safe_index)

        # Mask out invalid time steps
        cloud_base_height_asl = cloud_base_height_asl.where(is_valid)
        cloud_base_height_agl = cloud_base_height_asl - X["altitude"]

        cloud_flag = (cloud_layers > 0).any(dim="height")

        low_cloud_flag = cloud_base_height_agl < 2000.0
        cloud_flag = cloud_flag.drop_vars(["range", "height"], errors="ignore")
        low_cloud_flag = low_cloud_flag.drop_vars(
            ["range", "height"], errors="ignore"
        )
        cloud_base_height_asl = cloud_base_height_asl.drop_vars(
            ["range", "height"], errors="ignore"
        )
        cloud_base_height_agl = cloud_base_height_agl.drop_vars(
            ["range", "height"], errors="ignore"
        )
        fog_flag = X["fog_flag"].drop_vars(
            ["range", "height"], errors="ignore"
        )

        # Remove profiles with detected clouds from fog detection, as these
        # are likely false positives for fog (e.g. the cloud triggers the detection).
        fog_flag = fog_flag & ~low_cloud_flag
        low_cloud_and_fog_flag = low_cloud_flag | fog_flag
        low_cloud_flag = low_cloud_flag.where(~fog_flag, other=np.nan)
        
        product.data = xr.Dataset(
            {
                "cloud_layer_mask": cloud_mask,
                "cloud_flag": cloud_flag,
                "low_cloud_flag": low_cloud_flag,
                "low_cloud_and_fog_flag": low_cloud_and_fog_flag,
                "cloud_base_height_asl": cloud_base_height_asl,
                "cloud_base_height_agl": cloud_base_height_agl,
                "fog_flag": fog_flag,
            }
        )

        product.data = set_variable_metadata(product.data)

        product.data.attrs["data_source"] = data_file.data_source
        product.data.attrs["site_id"] = data_file.site_id
        product.data.attrs["date"] = data_file.date.isoformat()
        product.data.attrs["data_file"] = str(data_file.data_path)
        instrument = data_file.instrument
        if instrument is not None:
            product.data.attrs["instrument"] = (
                f"{instrument.manufacturer} {instrument.name}"
            )
        else:
            product.data.attrs["instrument"] = "Unknown"

        product.data.attrs["creation_date"] = (
            datetime.datetime.now().isoformat()
        )

        product.data = product.data.sortby("time")

        product.data = product.data.drop_duplicates("time")

        start_of_day = pd.Timestamp(data_file.date).normalize()
        end_of_day = start_of_day + pd.Timedelta(days=1)
        product.data = product.data.sel(time=slice(start_of_day, end_of_day))

        return product

    def to_netcdf(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_netcdf(
            path,
            mode="w",
        )

    def get_raw_time_series_data(self) -> Dataset:
        """Return all native-sample variables defined only on the time axis."""
        return extract_raw_time_series_data(self.data)

    @property
    def data(self) -> Dataset:
        """Get the data associated with this product."""
        if self._data is None:
            raise ValueError("Product not computed.")
        return self._data

    @data.setter
    def data(self, data: xr.Dataset) -> None:
        """Set the data associated with this data file."""
        self._data = data
