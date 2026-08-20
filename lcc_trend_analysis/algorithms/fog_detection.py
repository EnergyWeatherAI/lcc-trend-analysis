from dataclasses import dataclass
from typing import Optional

import xarray as xr

from ..type_aliases import Dataset, DataArray


@dataclass
class CeilometerFogDetectionParameters:
    """Parameters for CeilometerogDetectionTransformer.
    Defaults are CloudnetPy defaults.

    Attributes:
        fog_filter_n_gates_for_signal_sum (int, optional): Number of lowest gates to sum for signal detection. Defaults to 20.
        fog_filter_signal_sum_threshold (float, optional): Threshold for signal sum to identify fog. Defaults to 2e-3.
        fog_filter_variance_threshold (float, optional): Minimum variance threshold for fog detection. Defaults to 1e-15.
        fog_filter_above_peak_threshold (float, optional): Threshold for cleaning values above peak. Defaults to 1e-5.
    """

    fog_filter_n_gates_for_signal_sum: float = 20
    fog_filter_signal_sum_threshold: float = 1e-3
    fog_attennuation_range: float = 250.0
    fog_max_attennuated_threshold: float = 3e-7
    fog_signal_value_threshold: float = 1e-5


class CeilometerFogDetectionTransformer:
    def __init__(
        self,
        params: Optional[CeilometerFogDetectionParameters] = None,
    ):
        """Initialize the transformer with filtering parameters.

        Args:
            params (CeilometerFogDetectionParameters | None): Parameters for fog detection
        """
        if params is None:
            params = CeilometerFogDetectionParameters()

        self.params: CeilometerFogDetectionParameters = params

    def transform(
        self,
        data: Dataset,
        var_for_detection: str = "beta_raw",
        method: str = "cloudnetpy",
    ) -> Dataset:
        """Mask fog profiles using following CloudnetPy's approach.

        Implements CloudnetPy's fog detection using _find_fog_profiles() method.

        Args:
            data (Dataset): Input dataset containing raw backscatter data
            with key specified by `var_for_detection`. The data must be range-corrected.

        Returns:
            Dataset: Dataset with filtered backscatter data and fog detection status.
        """

        assert method in ["cloudnetpy", "tuononen"], (
            "Invalid method. Choose 'cloudnetpy' or 'tuononen'."
        )

        beta_key = (
            var_for_detection
            if var_for_detection in data.data_vars
            else "beta"
        )
        beta = data[beta_key].copy()

        # Check if the input raw backscatter data has been range-corrected
        if "range_corrected" not in beta.attrs:
            raise ValueError(
                f"Input data must have 'range_corrected' attribute in '{beta_key}'."
            )

        if not beta.attrs["range_corrected"]:
            raise ValueError(
                "Input backscatter data must be range-corrected before fog detection."
            )

        beta = beta.fillna(0.0)
        # Fog detectiion from Tuononen et al. (2019)
        if method == "tuononen":
            is_fog_profile: DataArray = (
                beta.isel(height=slice(0, 2)).max(dim="height")
                > self.params.fog_signal_value_threshold
            ) & (
                beta.isel(
                    height=xr.ufuncs.fabs(
                        data["range"] - self.params.fog_attennuation_range
                    ).argmin(dim="height")
                )
                < self.params.fog_max_attennuated_threshold
            )
        elif method == "cloudnetpy":
            # CloudnetPy's fog detection based on signal sum over the first N gates
            signal_sum: DataArray = xr.ufuncs.abs(
                beta.isel(
                    height=slice(
                        0, self.params.fog_filter_n_gates_for_signal_sum
                    )
                ).sum(dim="height")
            )

            is_fog_profile = (
                signal_sum > self.params.fog_filter_signal_sum_threshold
            )

        data["fog_flag"] = is_fog_profile

        return data
