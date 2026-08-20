from dataclasses import dataclass, field

from ..type_aliases import Datetime, Timedelta
from .ceilometers import CeilometerDataFile
from .utils import DimensionalityError, FileContentError

@dataclass
class NIWADataFile(CeilometerDataFile):
    """A class to wrap NIWA CL61 and CL31 data files.

    Raises:
        DimensionalityError: Data file has invalid or duplicate time dimension.
        DimensionalityError: Data file does not contain height variable.
    """

    data_source: str = field(
        init=False,
        default="NIWA - Earth Sciences New Zealand",
    )

    def standardize_units_and_coordinates(self):
        if "time" not in self.data.sizes.keys() or self.data["time"].size == 0:
            raise DimensionalityError(
                f"No time dimension found in dataset {self.data_path}."
            )
            
        if "beta_att" in self.data.data_vars:
            self.data = self.data.rename_vars({"beta_att": "beta_raw"})

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
            
        self.instrument.calibration_coefficient = (
            1.0 # NIWA data is precalibrated using CloudnetPy
        )
        
        self.instrument.scaling_factor = (
                1.0  # NIWA data is already scaled
            )

    def set_attributes(self):
        super().set_attributes()

        self.data["beta_raw"].attrs["range_corrected"] = 1
        if "beta" in self.data.data_vars:
            self.data["beta"].attrs["range_corrected"] = 1

    def set_instrument(self) -> None:
        self.instrument = self.find_ceilometer_by_attr("title")
