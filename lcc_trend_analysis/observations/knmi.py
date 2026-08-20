from dataclasses import dataclass, field

import numpy as np

from ..type_aliases import Datetime, Timedelta
from .ceilometers import CHM15k, CHM15kx, CT75k, CeilometerDataFile
from .utils import DimensionalityError, FileContentError

original_calibration_coefficients = {
    "cabauw": 0.301401,
}

@dataclass
class KNMIDataFile(CeilometerDataFile):
    """A class to wrap KNMI CHM15k and CT75k data files.

    Raises:
        DimensionalityError: Data file has invalid or duplicate time dimension.
        DimensionalityError: Data file does not contain height variable.
    """

    data_source: str = field(
        init=False,
        default="KNMI - Koninklijk Nederlands Meteorologisch Instituut",
    )

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

        if isinstance(self.instrument, CT75k):
            self.instrument.scaling_factor = (
                1.0  # KNMI CT75k data is already scaled
            )

        """
        if (
            isinstance(self.instrument, (CHM15k, CHM15kx))
            and "scaling" in self.data.data_vars
        ):
            # Revert instrument internal scaling/calibration back to original calibration.
            internal_calibration_factor = self.data["scaling"].median().item()
            self.instrument.calibration_coefficient = (
                internal_calibration_factor
                / original_calibration_coefficients.get(
                    self.site_id,
                    self.instrument.cloudnet_calibration_coefficient,
                )
            )
        """

    def set_attributes(self):
        super().set_attributes()

        self.data["beta_raw"].attrs["range_corrected"] = 1
        if "beta" in self.data.data_vars:
            self.data["beta"].attrs["range_corrected"] = 1

    def set_instrument(self) -> None:
        self.instrument = self.find_ceilometer_by_attr("title")
