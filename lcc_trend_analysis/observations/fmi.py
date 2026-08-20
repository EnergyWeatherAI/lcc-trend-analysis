from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..type_aliases import Dataset, Datetime, Timedelta
from .ceilometers import CeilometerDataFile
from .utils import DimensionalityError, FileContentError


@dataclass
class FMIDataFile(CeilometerDataFile):
    """A class to wrap FMI data files.

    Raises:
        DimensionalityError: Data file has invalid or duplicate time dimension.
        DimensionalityError: Data file does not contain height variable.
    """

    data_source: str = field(
        init=False,
        default="FMI - Finnish Meteorological Institute",
    )

    def load_data(self, engine="netcdf4") -> Dataset:
        ds: Dataset = xr.open_dataset(
            self.data_path, engine=engine, chunks=None, decode_cf=False
        )
        ds = self._fix_attrs(ds)
        ds["time"] = ds["time"] * 3600  # Decimal hours -> seconds
        ds["time"].attrs["units"] = (
            f"seconds since {self.date.strftime('%Y-%m-%d')}T00:00:00Z"
        )
        ds = xr.decode_cf(ds)
        return ds

    def standardize_units_and_coordinates(self):
        if "time" not in self.data.sizes.keys() or self.data["time"].size == 0:
            raise DimensionalityError(
                f"No time dimension found in dataset {self.data_path}."
            )

        if "beta_raw" not in self.data.data_vars:
            raise FileContentError(
                f"No 'beta_raw' variable found in dataset {self.data_path}."
            )

        if self.data["beta_raw"].dims.count("time") != 1:
            raise DimensionalityError(
                f"Data file {self.data_path} has invalid or duplicate time dimension, which is not supported."
            )

        if "time" in self.data.coords:
            self.start_time: Datetime = self.data.time[0].values
            self.end_time: Datetime = self.data.time[-1].values
            self.duration: Timedelta = self.end_time - self.start_time

        # On some files "elev" is the azimuth angle, not the elevation angle.
        if np.median(self.data.get("elev", 0.0)) < 45.0:
            self.data["elev"] = 90 - self.data.get("elev", 0.0)

    def set_attributes(self):
        super().set_attributes()
        
        self.data["beta_raw"].attrs["range_corrected"] = 1
        if "beta" in self.data.data_vars:
            self.data["beta"].attrs["range_corrected"] = 1
            
    def pre_calibration(self):
        self.instrument.scaling_factor = 1.0  # FMI data is already scaled

    def set_instrument(self) -> None:
        # Lookup instrument based on file name
        self.instrument = self.find_ceilometer_by_name(self.data_path.stem)
        