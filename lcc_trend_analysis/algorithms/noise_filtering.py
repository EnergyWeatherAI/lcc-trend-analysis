import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr

from ..type_aliases import DataArray, Dataset
from .range_correction import CeilometerRangeCorrectionTransformer


@dataclass
class CeilometerNoiseParameters:
    """Instrument-dependent noise parameters."""
    # For the purpose of analysing liquid clouds, the default minimum noise level can be kept
    # relatively high to avoid excessive sensitivity to very low backscatter values. This would
    # limit the applicability of noise-screened data for e.g. aerosol detection,
    # but is acceptable here.
    noise_min: Optional[float] = 1e-9
    noise_smooth_min: Optional[float] = 4e-9


@dataclass
class CeilometerFilteringParameters:
    """Parameters for CeilometerFilteringTransformer.
    Defaults are CloudnetPy defaults.

    Attributes:
        noise_top_gate_fraction (float): Fraction of top gates used for noise estimation.
        noise_clean (float): Noise level to set for cleaned profiles.
        negative_filter_min_negative_gates (int): Minimum number of consecutive negative gates to trigger cleaning.
        negative_filter_skip_lowest_n_gates (int): Number of lowest gates to skip in negative filtering.
    """

    noise_top_gate_fraction: float = 0.1
    noise_clean: float = 4e-9

    negative_filter_min_negative_gates: int = 5
    negative_filter_skip_lowest_n_gates: int = 5
    negative_filter_max_gates: int = 95
    negative_filter_threshold: float = 8e-6

    fog_filter_above_peak_threshold: float = 2e-6

    snr_threshold: float = 5.0


class CeilometerFilteringTransformer:
    """Stateless transformer to filter raw lidar backscatter data.

    Transforms raw ceilometer backscatter signal into filtered backscatter
    using standard Cloudnet processing scheme."""

    def __init__(
        self,
        instrument_noise_params: Optional[CeilometerNoiseParameters] = None,
        filtering_params: Optional[CeilometerFilteringParameters] = None,
        smoothed: bool = False
    ):
        """Initialize the transformer with filtering parameters.

        Args:
            params (CeilometerFilteringParameters | None): Parameters for filtering
        """
        if filtering_params is None:
            filtering_params = CeilometerFilteringParameters()
        if instrument_noise_params is None:
            instrument_noise_params = CeilometerNoiseParameters()

        self.params: CeilometerFilteringParameters = filtering_params
        self.noise_params: CeilometerNoiseParameters = instrument_noise_params
        self.smooth = smoothed

    def _filter_negatives(
        self, beta: DataArray
    ) -> tuple[DataArray, DataArray]:
        """Filter negatives and low values above consequent negatives.

        Args:
            beta (DataArray): Input backscatter data

        Returns:
            tuple[DataArray, DataArray]: Filtered data and temporal mask of filtered profiles
        """

        def cumsumr(arr):
            out = np.zeros_like(arr, dtype=int)
            if len(arr) > 0:
                out[0] = int(arr[0])
                for i in range(1, len(arr)):
                    out[i] = out[i - 1] + 1 if arr[i] else 0
            return out

        # Find negative values within a pre-defined range.
        negatives: DataArray = beta < 0
        # Set the lowest n gates and gates above max_gates to NaN
        mask = np.ones(beta.sizes["height"], dtype=bool)
        mask[: self.params.negative_filter_skip_lowest_n_gates] = False
        mask[
            self.params.negative_filter_skip_lowest_n_gates
            + self.params.negative_filter_max_gates :
        ] = False
        try:
            negatives = negatives.where(mask, other=False)
        except ValueError:
            #print(mask)
            #print(negatives)
            exit(1)

        n_consequent_negatives = xr.apply_ufunc(
            cumsumr,
            negatives,
            input_core_dims=[["height"]],
            output_core_dims=[["height"]],
            vectorize=True,
            output_dtypes=[np.int32],
        )

        # Find where consecutive negatives exceed threshold.
        negatives_exceeds_threshold = (
            n_consequent_negatives
            > self.params.negative_filter_min_negative_gates
        )

        # For each time, set all heights above the first True in exceeds_threshold to True
        # cumsum along height: after first True, cumsum > 0
        removal_mask = negatives_exceeds_threshold.cumsum(dim="height") > 0

        cleaned_beta = beta.where(
            ~removal_mask | (beta > self.params.negative_filter_threshold)
        )
        cleaned_negative_profiles = removal_mask.any(dim="height")

        return cleaned_beta, cleaned_negative_profiles

    def _filter_fog(self, beta: DataArray, fog_mask: DataArray) -> DataArray:
        """Filter fog profiles by setting all values in fog profiles to NaN.

        Args:
            beta (DataArray): Input backscatter data
            fog_mask (DataArray): Boolean mask indicating fog profiles

        Returns:
            DataArray: Filtered backscatter data
        """

        peak_ind = beta.argmax(dim="height", skipna=True)

        # Create height index array for broadcasting
        height_indices = xr.DataArray(
            np.arange(beta.sizes["height"]),
            dims=["height"],
            coords={"height": beta.height},
        )

        # Create mask: fog profiles AND above peak AND below threshold
        above_peak = height_indices > peak_ind  # type: ignore
        below_threshold = beta < self.params.fog_filter_above_peak_threshold
        mask_to_nan = fog_mask & above_peak & below_threshold

        # Apply the mask to set values to NaN
        beta = beta.where(~mask_to_nan)

        return beta

    def _filter_noise(self, beta: DataArray, noise: DataArray) -> DataArray:
        """Filter noise using SNR threshold.

        Args:
            beta (DataArray): Input backscatter data
            noise (DataArray): Background noise estimate

        Returns:
            DataArray: SNR-filtered backscatter data
        """

        snr = beta / noise

        beta = beta.where(snr >= self.params.snr_threshold)

        return beta

    def _filter(
        self,
        beta: DataArray,
        beta_raw: DataArray,
        range_var: DataArray,
        fog_flag: DataArray,
        filter_negatives: bool = True,
        filter_fog: bool = False,
        filter_snr: bool = True,
        disable_noise_level_estimation: bool = False,
    ) -> tuple[DataArray, DataArray, DataArray]:
        # Check if the input raw backscatter data has been range-corrected
        if "range_corrected" not in beta.attrs:
            raise ValueError(
                "Input data must have 'range_corrected' attribute in 'beta_raw'."
            )

        if not beta.attrs["range_corrected"]:
            raise ValueError(
                "Input raw backscatter data must be range-corrected before filtering."
            )

        # For filtering process, we need the non-range-corrected beta data.
        range_correction_transformer = CeilometerRangeCorrectionTransformer()
        filtered_beta = range_correction_transformer.inverse_transform(
            beta, range_var=range_var   
        )
        filtered_beta_raw = range_correction_transformer.inverse_transform(
            beta_raw, range_var=range_var
        )

        # 1. Estimate background variance and (adjusted) noise from beta_raw
        n_gates = round(
            len(filtered_beta_raw.height) * self.params.noise_top_gate_fraction
        )
        top_gates = filtered_beta_raw.isel(height=slice(-n_gates, None))
        background_variance = top_gates.var(dim="height", skipna=True)
        background_noise = xr.ufuncs.sqrt(background_variance)
        background_noise_original = background_noise.copy() # Original before min thresholding
        background_noise = background_noise.where(
            background_noise < self.noise_params.noise_min,
            self.noise_params.noise_min,
        )
        if disable_noise_level_estimation:
            background_noise[:] = self.noise_params.noise_min

        # 2. Filter profiles where consecutive negative values are detected.
        if filter_negatives:
            filtered_beta, cleaned_negative_profiles = self._filter_negatives(
                beta=filtered_beta
            )
            # background_noise[cleaned_negative_profiles] = (
            #     self.params.noise_clean
            # )

        # 3. Detect and filter fog profiles.
        if filter_fog:
            filtered_beta = self._filter_fog(
                beta=filtered_beta, fog_mask=fog_flag
            )
            # background_noise[data["fog_flag"]] = self.params.noise_clean

        # 4. Filter noise using SNR threshold
        if filter_snr:
            filtered_beta = self._filter_noise(
                beta=filtered_beta, noise=background_noise
            )

        if np.all(np.isnan(filtered_beta)):
            warnings.warn(
                f"All-NaN data backscatter data in data file after filtering. Median background noise: {background_noise.median():.2e}.",
                RuntimeWarning,
            )

        # Range-correct the filtered data
        filtered_beta = range_correction_transformer.transform(
            filtered_beta, range_var=range_var
        )

        return filtered_beta, background_noise_original, cleaned_negative_profiles

    def filter(
        self,
        data: Dataset,
        var_for_screening: str = "beta_raw",
        filter_negatives: bool = True,
        filter_fog: bool = False,
        filter_snr: bool = True,
        disable_noise_level_estimation: bool = False,
    ) -> Dataset:
        """Applies Cloudnet filtering to the input data.

        Args:
            data (Dataset): Input dataset containing raw backscatter data with key "beta_raw"
            var_for_screening (str): Variable to use for screening, either "beta_raw" or "beta_smooth"
            filter_negatives (bool): Whether to filter negative profiles
            filter_fog (bool): Whether to filter fog profiles
            filter_snr (bool): Whether to apply SNR threshold filtering
            disable_noise_level_estimation (bool): Whether to disable noise level estimation and use fixed minimum noise level instead

        Returns:
            Dataset: Dataset with filtered backscatter data and fog detection status
        """
        # Apply filtering filters in sequence following CloudnetPy approach

        # Create a copy of the raw beta data to avoid modifying the original
        assert "beta_raw" in data.data_vars, (
            "Input data must contain 'beta_raw' variable."
        )
        assert "range" in data.coords, (
            "Input data must contain 'range' coordinate for filtering."
        )

        filtered_beta_var, background_noise, cleaned_negative_profiles = self._filter(
            data[var_for_screening].copy(),
            data["beta_raw"].copy(),
            range_var=data["range"],
            fog_flag=data["fog_flag"],
            filter_negatives=filter_negatives,
            filter_fog=filter_fog,
            filter_snr=filter_snr,
            disable_noise_level_estimation=disable_noise_level_estimation,
        )

        
        filtered_beta_var.attrs["filtering"] = (
            f"SNR threshold applied: {self.params.snr_threshold}. Minimum background noise level: {self.noise_params.noise_min}."
        )
        
        # Mask the beta variable with the same mask as beta_var if not already beta_raw
        if var_for_screening != "beta_raw":
            
            filtered_beta = data["beta_raw"].copy().where(~filtered_beta_var.isnull())
            filtered_beta = filtered_beta.where(~(filtered_beta < 0.))
            
            data["beta"] = filtered_beta
            data[var_for_screening] = filtered_beta_var
            data[var_for_screening].attrs["filtering"] = (
                f"SNR threshold applied: {self.params.snr_threshold}. Minimum background noise level: {self.noise_params.noise_min}."
            )
        else:
            data["beta"] = filtered_beta_var
    
        data["beta"].attrs["filtering"] = (
                f"SNR threshold applied: {self.params.snr_threshold}. Minimum background noise level: {self.noise_params.noise_min}."
            )
        
        data["background_noise"] = background_noise

        if filter_negatives:
            data["cleaned_negative_profile_flag"] = cleaned_negative_profiles

        return data
