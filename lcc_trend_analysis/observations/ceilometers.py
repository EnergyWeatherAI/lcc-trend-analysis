import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, Self

import pandas as pd
import xarray as xr
import numpy as np

from ..metadata import set_variable_metadata
from ..algorithms.fog_detection import CeilometerFogDetectionTransformer
from ..algorithms.noise_filtering import (
    CeilometerFilteringTransformer,
    CeilometerNoiseParameters,
)
from ..algorithms.gaussian_smoothing import (
    BackscatterGaussianSmoothingTransformer,
)
from ..algorithms.range_correction import CeilometerRangeCorrectionTransformer
from ..type_aliases import Dataset, Timestamp

HOSTNAME: str = os.environ.get("HOSTNAME", "unknown_host")
USER: str = os.environ.get("USER", "unknown_user")


@dataclass
class Ceilometer(ABC):
    name: ClassVar[str]
    manufacturer: ClassVar[str]
    alternative_names: ClassVar[set[str]] = set()
    noise_params: CeilometerNoiseParameters = field(
        default_factory=CeilometerNoiseParameters
    )
    cloudnet_calibration_coefficient: float = field(default=1.0)
    calibration_coefficient: float = field(default=1.0)
    scaling_factor: float = field(default=1.0)


@dataclass
class GenericCeilometer(Ceilometer):
    name = "Unknown ceilometer"
    manufacturer = ""


@dataclass
class CHM15k(Ceilometer):
    name = "CHM15k"
    manufacturer = "Lufft"
    alternative_names = {"CHM15x", "CHM15kx"}
    wavelength = 1064e-9
    # cloudnet_calibration_coefficient: float = field(default=0.3)
    scaling_factor: float = field(default=1e-11)
    stratocumulus_lidar_ratio: float = 18.2


@dataclass
class CHM15kx(Ceilometer):
    name = "CHM15kx"
    manufacturer = "Lufft"
    wavelength = 1064e-9
    # cloudnet_calibration_coefficient: float = field(default=0.3)
    scaling_factor: float = field(default=1e-11)
    stratocumulus_lidar_ratio: float = 18.2


@dataclass
class LD40(Ceilometer):
    name = "LD40"
    manufacturer = "Vaisala"
    wavelength = 855e-9
    cloudnet_calibration_coefficient: float = field(default=2.45)


@dataclass
class CL31(Ceilometer):
    name = "CL31"
    manufacturer = "Vaisala"
    wavelength = 910e-9
    cloudnet_calibration_coefficient: float = field(default=1.45)
    scaling_factor: float = field(default=1e-8)
    stratocumulus_lidar_ratio: float = 18.8


@dataclass
class CL51(Ceilometer):
    name = "CL51"
    manufacturer = "Vaisala"
    wavelength = 910e-9
    cloudnet_calibration_coefficient: float = field(default=1.2)
    scaling_factor: float = field(default=1e-7)
    stratocumulus_lidar_ratio: float = 18.8


@dataclass
class CL61(Ceilometer):
    name = "CL61"
    alternative_names = {"CL61d"}
    manufacturer = "Vaisala"
    wavelength = 910.55e-9
    scaling_factor: float = field(default=1.0)
    stratocumulus_lidar_ratio: float = 18.8


@dataclass
class CT25k(Ceilometer):
    name = "CT25k"
    manufacturer = "Vaisala"
    wavelength = 905e-9
    cloudnet_calibration_coefficient: float = field(default=1.2)
    scaling_factor: float = field(default=1e-7)
    stratocumulus_lidar_ratio: float = 18.8


@dataclass
class CT75k(Ceilometer):
    name = "CT75k"
    manufacturer = "Vaisala"
    alternative_names = {"CT75"}
    wavelength = 905e-9
    cloudnet_calibration_coefficient: float = field(default=2.45)
    scaling_factor: float = field(default=1e-7)
    stratocumulus_lidar_ratio: float = 18.8


@dataclass
class PollyXT(Ceilometer):
    name = "PollyXT"
    manufacturer = "TROPOS"
    wavelength = 1064e-9


@dataclass
class EZLidar(Ceilometer):
    name = "EZLidar"
    manufacturer = "Leosphere"
    wavelength = 355 - 9


@dataclass
class DA10(Ceilometer):
    name = "DA10"
    manufacturer = "Vaisala"
    wavelength = 910e-9
    calibration_coefficient: float = field(default=1.0)
    scaling_factor: float = field(default=1.0)


CEILOMETERS: list = [
    CL61,
    CHM15kx,
    CHM15k,
    CL51,
    CL31,
    CT25k,
    CT75k,
    LD40,
    PollyXT,
    DA10,
    EZLidar,
]

PRODUCT_VARS: list[str] = [
    "beta",
    "beta_raw",
    "fog_flag",
    "range",
    "altitude",
]

STATE_VARS: list[str] = [
    "background_noise",
    "cleaned_negative_profile_flag",
    "scaling",
    "c_cal",
    "state_detector",
    "state_laser",
    "state_optics",
    "receiver_sens",
    "receiver_sensitivity",
    "laser_energy",
    "window_contamination",
    "window_transmission",
    "laser_pulse_energy",
    "qc_background_light",
    "qc_laser_pulse_energy",
    "qc_laser_temperature",
    "qc_window_transmission",
]


@dataclass
class CeilometerDataFile(ABC):
    """A generic class to wrap ceilometer data files.

    Attributes:
        date (Timestamp): The date of the data file.
        site_id (str): The site identifier.
        data_path (Path): The path to the data file.
    """

    site_id: str
    site_name: str
    date: Timestamp
    data_path: Path
    data_source: str = field(default="Unknown")
    preprocessed: bool = field(default=False)
    instrument: Optional[Ceilometer] = field(default=None)
    _data: Optional[Dataset] = field(default=None, repr=False)

    def _fix_attrs(self, ds: Dataset) -> Dataset:
        for var in ds.variables:
            if {"_FillValue", "missing_value"} <= ds[var].attrs.keys():
                ds[var].attrs.pop("missing_value", None)
            elif "missing_value" in ds[var].attrs:
                ds[var].attrs["_FillValue"] = ds[var].attrs["missing_value"]
        return ds

    def find_ceilometer_by_attr(self, lookup_attr: str) -> Ceilometer:
        lookup_string = self.data.attrs.get(lookup_attr, None)
        ceilometer: Ceilometer | None = None
        if lookup_string is not None:
            ceilometer = self.find_ceilometer_by_name(lookup_string)
            if not isinstance(ceilometer, GenericCeilometer):
                return ceilometer

        if "beta" in self.data.data_vars:
            lookup_string = self.data["beta"].attrs.get(lookup_attr, None)

        if lookup_string is not None:
            ceilometer = self.find_ceilometer_by_name(lookup_string)

        ceilometer = ceilometer or GenericCeilometer()
        if isinstance(ceilometer, GenericCeilometer):
            warnings.warn(
                f"Could not find a specific ceilometer match for lookup attribute '{lookup_attr}' in data file {self.data_path}. Defaulting to generic ceilometer."
            )

        return ceilometer

    def _km_to_m(self, var_name: str) -> None:
        if self.data[var_name].attrs.get("units", None) == "km":
            attrs = self.data[var_name].attrs
            self.data = self.data.assign_coords(
                {var_name: self.data[var_name] * 1000}
            )
            self.data[var_name].attrs = attrs
            self.data[var_name].attrs["units"] = "m"

    def _angle_correction(self) -> None:
        angle_var = (
            "tilt_angle" if "tilt_angle" in self.data.data_vars else "azimuth"
        )
        if angle_var in self.data.data_vars:
            attrs = self.data["height"].attrs
            self.data = self.data.assign_coords(
                height=self.data["height"]
                * np.cos(np.deg2rad(np.median(self.data.get(angle_var, 0.0))))
            )
            self.data["height"].attrs = attrs
        elif "elev" in self.data.data_vars:
            attrs = self.data["height"].attrs
            self.data = self.data.assign_coords(
                height=self.data["height"]
                * np.sin(np.deg2rad(np.median(self.data.get("elev", 90.0))))
            )
            self.data["height"].attrs = attrs

    def _set_altitude(self) -> None:
        if "alt" in self.data.data_vars:
            self.data = self.data.rename({"alt": "altitude"})
        elif "station_altitude" in self.data.data_vars:
            self.data = self.data.rename({"station_altitude": "altitude"})
        if "altitude" not in self.data.data_vars:
            warnings.warn(
                f"Altitude data is missing in data file {self.data_path}. Using 0 m as default, which may lead to incorrect height values."
            )
            self.data["altitude"] = 0.0
            self.data["altitude"].attrs["units"] = "m"

        if self.data["altitude"].size > 1:
            self.data["altitude"] = self.data["altitude"].median().squeeze()

    def fix_height_and_range_coordinates(self) -> None:
        """Ensure height and range coordinates are present and consistent."""
        range_coord_present: bool = "range" in self.data.coords
        height_coord_present: bool = "height" in self.data.coords
        self._set_altitude()
        if "height" in self.data.data_vars and range_coord_present:
            self._km_to_m("height")
            self._km_to_m("range")
            range: xr.DataArray = self.data["range"]
            range_attrs = range.attrs
            self.data = self.data.assign_coords(height=self.data["height"])
            self.data = self.data.rename_dims({"range": "_old_range"})
            self.data = self.data.swap_dims({"_old_range": "height"})
            self.data = self.data.assign_coords(range=("height", range.values))
            self.data["range"].attrs = range_attrs

        elif range_coord_present and not height_coord_present:
            self._km_to_m("range")
            range_vals = self.data["range"].data
            self.data = self.data.rename_dims({"range": "height"})

            self.data = self.data.assign_coords({"height": range_vals})

            self._angle_correction()

            self.data = self.data.assign_coords(
                height=self.data["height"] + self.data["altitude"]
            )

        elif range_coord_present and height_coord_present:
            self._km_to_m("range")
            self._km_to_m("height")

        elif height_coord_present and not range_coord_present:
            self._km_to_m("height")
            self.data = self.data.assign_coords(
                {"range": self.data["height"] - self.data["altitude"]}
            )

        if "scalar" in self.data.dims:
            self.data = self.data.squeeze("scalar", drop=True)

    @staticmethod
    def find_ceilometer_by_name(lookup_str: str) -> Ceilometer:
        for ceilometer_cls in CEILOMETERS:
            if ceilometer_cls.name.lower() in lookup_str.lower():
                return ceilometer_cls()
            for name in ceilometer_cls.alternative_names:
                if name.lower() in lookup_str.lower():
                    return ceilometer_cls()
        return GenericCeilometer()

    def calibrate(self) -> None:
        """Calibrate the backscatter data using instrument-specific parameters."""
        self.require_raw_backscatter()
        instrument = self.require_instrument()

        calibration_factor = (
            instrument.scaling_factor
            * instrument.calibration_coefficient
            * instrument.cloudnet_calibration_coefficient
        )

        with xr.set_options(keep_attrs=True):
            self.data["beta_raw"] *= calibration_factor
        self.data["beta_raw"].attrs["calibration_factor"] = calibration_factor

        # Store original scaling for CHM15k
        if "scaling" in self.data.data_vars:
            self.data["scaling"] = self.data["scaling"].median()

    def clear_empty_profiles(self) -> None:
        """Remove empty profiles from the data."""
        self.require_raw_backscatter()

        # Find time steps where all heights are NaN
        all_nan = self.data["beta_raw"].isnull().all(dim="height")
        nan_count = all_nan.sum()

        self.data = self.data.dropna(
            dim="time", how="all", subset=["beta_raw"]
        )

        if nan_count > 10:
            warnings.warn(
                f"Dropped more than 10 empty profiles (n={nan_count.item()}) from data file {self.data_path}."
            )

    def clear_empty_height_levels_from_top(self) -> None:
        """Remove completely empty heights, which can interfere noise detection, from the top of the profile."""
        self.require_raw_backscatter()
        # Find the topmost height index with non-all-nan/zero data.
        all_zero = (self.data["beta_raw"] == 0.0).all(dim="time")
        all_nan = self.data["beta_raw"].isnull().all(dim="time")
        all_nonvalid = all_zero | all_nan
        top_idx = (
            self.data["height"]
            .where(~all_nonvalid)
            .argmax(dim="height")
            .item()  # type: ignore
        )
        if top_idx < (self.data["height"].size - 11):
            warnings.warn(
                f"More than 10 empty top gates were dropped from data file {self.data_path}. Original top index: {self.data['height'].size - 1}, new top index: {top_idx}."
            )

        self.data = self.data.isel(height=slice(0, top_idx + 1))

    def screen_sunbeam(self) -> None:
        self.require_raw_backscatter()

        high_alt_mask: xr.DataArray = self.data["range"] > 10000.0
        if not high_alt_mask.any().item():
            return

        high_alt_backscatter = self.data["beta_raw"].where(high_alt_mask)
        n_heights = high_alt_backscatter.sizes.get("height", 0)
        if n_heights == 0:
            return

        n_bins = min(20, n_heights)
        bin_size = n_heights // n_bins

        binned_has_data = (
            high_alt_backscatter.notnull()
            .isel(height=slice(0, bin_size * n_bins))
            .coarsen(height=bin_size, boundary="trim")
            .max()
        )

        valid_profiles = binned_has_data.sum(dim="height") < 15
        self.data = self.data.sel(
            time=valid_profiles.time.where(valid_profiles, drop=True)
        )

    def detect_fog(self, var_for_detection: str = "beta_raw") -> None:
        """Apply fog detection to the backscatter data."""
        fog_detection_transformer = CeilometerFogDetectionTransformer()
        self.data = fog_detection_transformer.transform(
            self.data, var_for_detection=var_for_detection
        )

    def filter_backscatter(self, **kwargs) -> None:
        """Apply filtering to the backscatter data."""
        instrument = self.require_instrument()

        smoothed = kwargs.get("var_for_screening", "") == "beta_smooth"
        filter_transformer = CeilometerFilteringTransformer(
            instrument.noise_params, smoothed=smoothed
        )
        with warnings.catch_warnings(record=True) as emitted_warnings:
            warnings.simplefilter("always")
            self.data = filter_transformer.filter(self.data, **kwargs)

        if emitted_warnings:
            for warn in emitted_warnings:
                warnings.warn(
                    f"While noise filtering backscatter data from {self.data_path}, the following warning was issued:\n"
                    f"  {warn.message} ({warn.filename}:{warn.lineno})",
                    category=warn.category,
                )

    def load_data(self, engine="netcdf4") -> Dataset:
        ds: Dataset = xr.open_dataset(
            self.data_path, engine=engine, chunks=None, decode_cf=False
        )
        ds = self._fix_attrs(ds)
        ds = xr.decode_cf(ds)
        return ds

    @abstractmethod
    def standardize_units_and_coordinates(self) -> None:
        """Standardize units and coordinates for this data file."""
        pass

    @abstractmethod
    def set_instrument(self) -> None:
        """Set the instrument for this data file."""
        pass

    def set_attributes(self) -> None:
        """Set variable attributes for this data file."""
        self.require_raw_backscatter()

    def pre_calibration(self) -> None:
        """Perform any pre-calibration steps required for this data file."""
        pass

    def compute_smoothed_backscatter(self) -> None:
        """Compute Gaussian-smoothed backscatter profiles for noise screening."""
        smoothing_transformer = BackscatterGaussianSmoothingTransformer()

        with warnings.catch_warnings(record=True) as emitted_warnings:
            warnings.simplefilter("always")
            self.data = smoothing_transformer.smooth(self.data)

        if emitted_warnings:
            for warn in emitted_warnings:
                warnings.warn(
                    f"While smoothing backscatter data from {self.data_path}, the following warning was issued:\n"
                    f"  {warn.message} ({warn.filename}:{warn.lineno})",
                    category=warn.category,
                )

    @classmethod
    def from_source_data_file(
        cls, site_id: str, site_name: str, date: Timestamp, data_path: Path
    ) -> Self:
        """Create a CeilometerDataFile from a netCDF data file.

        Args:
            site_id (str): _description_
            site_name (str): _description_
            date (Timestamp): _description_
            data_path (Path): _description_

        Returns:
            Self: A CeilometerDataFile instance.
        """
        product: Self = cls(site_id, site_name, date, data_path)

        return product
    
    def drop_duplicate_time_steps(self) -> None:
        """Drop duplicate time steps from the data."""
        if "time" in self.data.coords:
            self.data = self.data.drop_duplicates(dim="time", keep="first")

    def preprocess(self) -> None:
        """Preprocess a ALC data file."""
        self.drop_duplicate_time_steps()
        
        self.standardize_units_and_coordinates()

        self.fix_height_and_range_coordinates()

        self.set_attributes()

        self.clear_empty_profiles()

        self.clear_empty_height_levels_from_top()

        self.range_correction()

        self.pre_calibration()

        self.calibrate()

        self.screen_sunbeam()

        self.detect_fog()

        # Smooth backscatter for modern instruments
        if isinstance(self.instrument, (CL61, CHM15k, CHM15kx, CL51, CL31)):
            self.compute_smoothed_backscatter()
            self.filter_backscatter(var_for_screening="beta_smooth")
        else:
            self.filter_backscatter(var_for_screening="beta_raw")

    def range_correction(self) -> None:
        """Apply range correction to backscatter data if not already applied."""
        for var in ["beta", "beta_raw"]:
            if var not in self.data.data_vars:
                continue
            if not self.data[var].attrs["range_corrected"]:
                range_correction = CeilometerRangeCorrectionTransformer()
                self.data[var] = range_correction.transform(
                    self.data[var], self.data["range"]
                )

    def to_netcdf_product(self, path: Path) -> None:
        """Save the data into a standardized L1b netCDF product file.

        This method stores a standard set of variables and attributes into a product file.

        Args:
            path (Path): The path to save the netCDF file.
        """
        instrument = self.require_instrument()

        # Subset data to only include product variables that are present
        available_product_vars = [
            var for var in PRODUCT_VARS if var in self.data.data_vars
        ]
        available_state_vars = [
            var for var in STATE_VARS if var in self.data.data_vars
        ]

        data_to_store = self.data[
            available_product_vars + available_state_vars
        ]

        for varname, var in data_to_store.data_vars.items():
            var.encoding = {}

        autocalibration_attrs = {
            key: self.data.attrs[key]
            for key in [
                "autocalibration_factor",
                "autocalibration_applied",
                "autocalibration_stage",
            ]
            if key in self.data.attrs
        }

        # Set global attributes
        data_to_store.attrs = {
            "title": f"Preprocessed ceilometer data from {self.site_name}",
            "instrument": f"{instrument.manufacturer} {instrument.name}",
            "wavelength": f"{getattr(instrument, 'wavelength', 0) * 1e9:.0f} nm",
            "institution": "TU Delft",
            "source": self.data_source,
            "history": f"Created by {USER} on {HOSTNAME} at {pd.Timestamp.now(tz='UTC').isoformat()}Z",
            "references": "",
            **autocalibration_attrs,
        }

        # Set variable attributes
        data_to_store = set_variable_metadata(data_to_store)

        data_to_store.to_netcdf(
            path,
            mode="w",
            engine="netcdf4",
        )

    @property
    def data(self) -> Dataset:
        """Get the data associated with this data file."""
        if self._data is None:
            self._data = self.load_data()
        return self._data

    @data.setter
    def data(self, data: Dataset) -> None:
        """Set the data associated with this data file."""
        self._data = data

    def require_raw_backscatter(self) -> None:
        """Validate that the file contains unscreened raw backscatter."""
        if not self.has_raw_data:
            raise ValueError(
                f"Raw backscatter variable 'beta_raw' is required for {self.data_path}."
            )

    def require_instrument(self) -> Ceilometer:
        """Return the resolved instrument metadata for this file."""
        if self.instrument is None:
            raise ValueError(
                f"Instrument metadata has not been resolved for {self.data_path}."
            )
        return self.instrument

    @property
    def has_raw_data(self) -> bool:
        """Check if the data file contains raw backscatter data."""
        return "beta_raw" in self.data.data_vars


class Level1bcCeilometerDataFile(CeilometerDataFile):
    """A subclass of CeilometerDataFile for already preprocessed data files."""

    def preprocess(self) -> None:
        """Preprocessing is not needed for already preprocessed data files."""
        pass

    def set_instrument(self):
        pass

    def standardize_units_and_coordinates(self):
        pass

    @classmethod
    def from_level1bc_data_file(
        cls,
        site_id: str,
        site_name: str,
        instrument: str,
        date: Timestamp,
        data_path: Path,
    ) -> Self:
        """Create a CeilometerDataFile instance from an already processed data file.

        Args:
            site_id (str): The site identifier.
            site_name (str): The site name.
            instrument (str): The instrument name.
            date (Timestamp): The date of the data file.
            data_path (Path): The path to the data file.

        Returns:
            Self: Processed CeilometerDataFile instance.
        """

        product: Self = cls(site_id, site_name, date, data_path)
        product.instrument = product.find_ceilometer_by_name(instrument)
        product.preprocessed = True

        return product
