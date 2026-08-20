from dataclasses import dataclass, field

import numpy as np

from ..type_aliases import Datetime, Timedelta
from .ceilometers import CeilometerDataFile
from .utils import DimensionalityError


@dataclass
class ARMDataFile(CeilometerDataFile):
    """A class to wrap ARM ceilometer data files.

    Raises:
        DimensionalityError: Data file does not contain altitude attribute.
        DimensionalityError: Data file has invalid or duplicate time dimension.
        DimensionalityError: Data file does not contain range variable.
    """

    data_source: str = field(
        init=False,
        default="Atmospheric Radiation Measurement (ARM) user facility; U.S. Department of Energy (DOE)",
    )

    def standardize_units_and_coordinates(self):
        if self.data["backscatter"].dims.count("time") != 1:
            raise DimensionalityError(
                f"Data file {self.data_path} has invalid or duplicate time dimension, which is not supported."
            )

        if "time" in self.data.coords:
            self.start_time: Datetime = self.data.time[0].values
            self.end_time: Datetime = self.data.time[-1].values
            self.duration: Timedelta = self.end_time - self.start_time

        self.data = self.data.rename({"backscatter": "beta_raw"})

    def set_attributes(self):
        super().set_attributes()

        if "range" not in self.data.coords:
            raise DimensionalityError(
                f"No range variable found in dataset {self.data_path}."
            )

        # ARM data is range corrected
        self.data["beta_raw"].attrs["range_corrected"] = 1

    def set_instrument(self) -> None:
        self.instrument = self.find_ceilometer_by_attr("ceilometer_model")
