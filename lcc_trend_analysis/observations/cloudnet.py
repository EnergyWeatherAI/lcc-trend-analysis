from dataclasses import dataclass, field

import pandas as pd
import xarray as xr

from .ceilometers import CL31, CeilometerDataFile
from ..type_aliases import Datetime, Timedelta
from .utils import DimensionalityError


@dataclass
class CloudnetDataFile(CeilometerDataFile):
    """A class to wrap Cloudnet data files.

    Raises:
        DimensionalityError: Data file has invalid or duplicate time dimension.
        DimensionalityError: Data file does not contain height variable.
    """

    data_source: str = field(
        init=False,
        default="ACTRIS Cloud remote sensing data centre unit (CLU)",
    )

    def standardize_units_and_coordinates(self):
        self.require_raw_backscatter()

        if self.data["beta_raw"].dims.count("time") != 1:
            raise DimensionalityError(
                f"Data file {self.data_path} has invalid or duplicate time dimension, which is not supported."
            )

        if "time" in self.data.coords:
            self.start_time: Datetime = self.data.time[0].values
            self.end_time: Datetime = self.data.time[-1].values
            self.duration: Timedelta = self.end_time - self.start_time

    def set_attributes(self) -> None:
        super().set_attributes()

        if self.site_id == "palaiseau" and self.data["range"].max() < 8e3:
            self.data["beta_raw"].attrs["range_corrected"] = 1
            if "beta" in self.data.data_vars:
                self.data["beta"].attrs["range_corrected"] = 1
        elif self.site_id == "chilbolton" and self.start_time < pd.Timestamp(
            "2014-12-11"
        ):
            self.data["beta_raw"].attrs["range_corrected"] = 0
            if "beta" in self.data.data_vars:
                self.data["beta"].attrs["range_corrected"] = 0
        elif (
            self.site_id == "cabauw"
            and self.start_time < pd.Timestamp("2012-01-01")
        ):
            # Cabauw CT75k 20001-2011 is not range-corrected, so we need to apply range correction.
            self.data["beta_raw"].attrs["range_corrected"] = 0
            if "beta" in self.data.data_vars:
                self.data["beta"].attrs["range_corrected"] = 0
        else:
            self.data["beta_raw"].attrs["range_corrected"] = 1
            if "beta" in self.data.data_vars:
                self.data["beta"].attrs["range_corrected"] = 1

    def pre_calibration(self):
        # Reverse CloudNet calibration to default instrument calibration
        instrument = self.require_instrument()

        if "calibration_factor" not in self.data.data_vars:
            inverse_calibration_factor = 1.0
        elif self.data["calibration_factor"].item() < 1e-4:
            # Calibration factor most likely includes the scaling factor
            inverse_calibration_factor = self.data["calibration_factor"].item() / instrument.scaling_factor
        elif self.site_id == "palaiseau" and isinstance(instrument, CL31):
            inverse_calibration_factor = 1.45
        else:
            inverse_calibration_factor = self.data["calibration_factor"].item()
        
        with xr.set_options(keep_attrs=True):
            self.data["beta_raw"] /= inverse_calibration_factor
        
        instrument.scaling_factor = 1.0  # Cloudnet data is already scaled
            

    def set_instrument(self) -> None:
        self.instrument = self.find_ceilometer_by_attr("source")
