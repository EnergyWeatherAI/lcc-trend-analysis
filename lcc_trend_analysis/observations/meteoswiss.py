from dataclasses import dataclass, field

from ..type_aliases import Datetime, Timedelta
from .ceilometers import CHM15kx, CeilometerDataFile, CHM15k
from .utils import DimensionalityError

original_calibration_coefficients = {
    "payerne": 0.428685,
}


@dataclass
class MeteoSwissCHM15kDataFile(CeilometerDataFile):
    """A class to wrap MeteoSwiss CHM15k data files.

    Raises:
        DimensionalityError: Data file has invalid or duplicate time dimension.
        DimensionalityError: Data file does not contain height variable.
    """

    data_source: str = field(
        init=False,
        default="MeteoSwiss - Swiss Federal Office of Meteorology and Climatology",
    )

    def standardize_units_and_coordinates(self):
        if self.data["beta_raw"].dims.count("time") != 1:
            raise DimensionalityError(
                f"Data file {self.data_path} has invalid or duplicate time dimension, which is not supported."
            )
        if "time" in self.data.coords:
            self.start_time: Datetime = self.data.time[0].values
            self.end_time: Datetime = self.data.time[-1].values
            self.duration: Timedelta = self.end_time - self.start_time

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

        # MeteoSwiss CHM15k data is range corrected
        self.data["beta_raw"].attrs["range_corrected"] = 1

        # Check instrument status
        self.data["beta_raw"] = self.data["beta_raw"].where(
            ~(
                (self.data["state_detector"] < 80.0)
                | (self.data["state_laser"] < 80.0)
                | (self.data["state_optics"] < 80.0)
            )
        )

    def set_instrument(self):
        self.instrument = CHM15k()
