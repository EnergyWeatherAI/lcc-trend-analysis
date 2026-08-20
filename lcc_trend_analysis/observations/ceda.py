from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from ..type_aliases import Dataset, Datetime, Timedelta
from .ceilometers import CeilometerDataFile, CT75k
from .utils import DimensionalityError


@dataclass
class Chilbolton_CT75k(CT75k):
    operator = "Science and Technology Facilities Council; Chilbolton Facility for Atmospheric and Radio Research"

@dataclass
class STFCChilboltonCT75kDataFile(CeilometerDataFile):
    """A class to wrap Vaisala CT75k data files from Chilbolton Facility for Atmospheric and Radio Research as obtained from CEDA.

    Raises:
        DimensionalityError: Data file does not contain altitude attribute.
        DimensionalityError: Data file has invalid or duplicate time dimension.
        DimensionalityError: Data file does not contain range variable.
    """

    data_source: str = field(
        init=False,
        default="Science and Technology Facilities Council; Chilbolton Facility for Atmospheric and Radio Research",
    )
    
    def load_data(self, engine="netcdf4") -> Dataset:
        ds: Dataset = xr.open_dataset(
            self.data_path, engine=engine, chunks=None, decode_cf=False
        )
        ds = self._fix_attrs(ds)
        
        ds = xr.decode_cf(ds)
        return ds

    def convert_time_dimension(self):
        """Convert time dimension from float (fractional hours) to datetime64 if needed."""

        if "time" not in self.data.coords:
            return

        time_coord = self.data.coords["time"]

        # Check if time is already datetime64
        if np.issubdtype(time_coord.dtype, np.datetime64):
            return

        # Check if time is float (fractional hours from self.date)
        if np.issubdtype(time_coord.dtype, np.floating):
            base_date = pd.to_datetime(self.date.strftime("%Y-%m-%d"))
            time_values = time_coord.values
            datetime_values = base_date + pd.to_timedelta(
                time_values, unit="h"
            )

            # Update the time coordinate
            self.data = self.data.assign_coords(time=datetime_values)

    def standardize_units_and_coordinates(self):
        if self.data["beta"].dims.count("time") != 1:
            raise DimensionalityError(
                f"Data file {self.data_path} has invalid or duplicate time dimension, which is not supported."
            )

        if "time" not in self.data.coords:
            raise DimensionalityError(
                f"Data file {self.data_path} is missing time dimension."
            )

        # Convert time dimension if it's in float format
        self.convert_time_dimension()

        self.start_time: Datetime = self.data.time[0].values
        self.end_time: Datetime = self.data.time[-1].values
        self.duration: Timedelta = self.end_time - self.start_time

    def set_attributes(self) -> None:
        super().set_attributes()
        
        if self.has_raw_data:
            self.data["beta_raw"].attrs["range_corrected"] = 1
        self.data["beta"].attrs["range_corrected"] = 1
        
    def pre_calibration(self):
        self.instrument.scaling_factor = 1.0  # CT75k data is already scaled

    def set_instrument(self):
        """Read calibration from "beta" attributes if available. Otherwise, use default for CT75k."""
        self.instrument = Chilbolton_CT75k()
