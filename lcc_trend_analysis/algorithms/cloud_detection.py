from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import scipy.signal
import xarray as xr

from ..type_aliases import DataArray, Dataset


@dataclass
class ALCCloudCoverEstimatorParameters:
    """Parameters for ALCCloudCoverEstimator.
    Defaults are based on CloudnetPy defaults.

    Attributes:
        min_peak_amplitude (float): Minimum amplitude of a peak to be considered.
        max_width (float): Maximum width of a peak in meters.
        min_width (float): Minimum width of a peak in meters.
        min_points (int): Minimum number of points in a peak to be considered.
        min_peak_top_beta_grad (float): Minimum gradient of beta at the peak top.
        min_alt (float): Minimum altitude of a peak in meters.
        peak_search_win_below (float): Search window distance below a local peak in meters.
        peak_search_win_above (float): Search window distance above a local peak in meters.
        peak_threshold_factor (float): Factor to threshold the peak gradient.
        min_peak_contrast_factor (float | None): Optional minimum contrast factor between the peak and values
            sampled above and below the peak. When None, this criterion is disabled.
        peak_contrast_distance_m (float | None): Optional vertical distance in meters for sampling above and below
            the peak in contrast checks. Used only when min_peak_contrast_factor is set.
    """

    min_peak_amplitude: float = 2e-5
    max_width: float = 300.0
    min_points: int = 3
    min_peak_top_beta_grad: float = 1e-7
    min_alt: float = 100.0
    peak_search_win_below: float = 200.0
    peak_search_win_above: float = 150.0
    peak_threshold_factor: float = 4.0
    min_peak_contrast_factor: float | None = None
    peak_contrast_distance_m: float | None = None


class ALCCloudCoverEstimator:
    """Stateless estimator to mask cloud layers from lidar backscatter data.

    Find cloud layers from lidar data using algorithm by
    Tuononen et al. (2019). This is a lidar-only modified version of
    the algorithm used in the Cloudnet processing framework.

    References:
        Implemented version of the algorithm:
        Tuononen et al., 2019: Evaluating solar radiation forecast uncertainty.
        Atmospheric Chemistry and Physics, 19, 1985-2000,
        https://doi.org/10.5194/acp-19-1985-2019

        Cloudnet implementation (with MWR-based LWP filtering):
        Tukiainen et al., 2020: CloudnetPy: A Python package for processing
        cloud remote sensing data. Journal of Open Source Software,
        5(53), 2123, https://doi.org/10.21105/joss.02123
    """

    def __init__(
        self,
        params: ALCCloudCoverEstimatorParameters = ALCCloudCoverEstimatorParameters(),
    ):
        """Initialize the cloud cover estimator.

        Args:
            params (CeilometerCloudCoverEstimatorParameters): Parameters for cloud detection
        """
        self.params: ALCCloudCoverEstimatorParameters = params

    def _is_qualified_peak(
        self,
        beta: np.ndarray,
        height: np.ndarray,
        top: int,
        base: int,
        peak_idx: int,
        min_height: float,
    ) -> bool:
        n_points: int = np.count_nonzero(beta[base : top + 1])
        peak_width: float = height[top] - height[base]
        peak_alt: float = (
            height[peak_idx] - min_height
        )  # Altitude of the peak above ground
        peak_top_beta_grad = (beta[peak_idx] - beta[top]) / (
            height[top] - height[peak_idx]
        )

        conditions: tuple = (
            n_points >= self.params.min_points,
            peak_width < self.params.max_width,
            peak_top_beta_grad > self.params.min_peak_top_beta_grad,
            peak_alt >= self.params.min_alt,
        )
        return all(conditions)

    def _passes_peak_contrast(
        self,
        beta_profile: np.ndarray,
        height: np.ndarray,
        peak_idx: int,
    ) -> bool:
        """Optional strict contrast criterion for high-confidence liquid cloud peaks."""
        if (
            self.params.min_peak_contrast_factor is None
            or self.params.peak_contrast_distance_m is None
        ):
            return True

        peak_height = height[peak_idx]
        below_target = peak_height - self.params.peak_contrast_distance_m
        above_target = peak_height + self.params.peak_contrast_distance_m

        below_candidates = np.where(height <= below_target)[0]
        above_candidates = np.where(height >= above_target)[0]

        if below_candidates.size == 0 or above_candidates.size == 0:
            return False

        below_idx = int(below_candidates[-1])
        above_idx = int(above_candidates[0])

        below_val = max(float(beta_profile[below_idx]), 0.0)
        above_val = max(float(beta_profile[above_idx]), 0.0)
        ref_val = max(below_val, above_val)

        peak_val = float(beta_profile[peak_idx])
        if ref_val <= 0:
            return peak_val > 0.0

        return peak_val >= self.params.min_peak_contrast_factor * ref_val

    def _find_local_extrema(
        self, array: np.ndarray, comparator: Callable, order: int
    ) -> np.ndarray:
        """Returns a mask of the local extrema in the array along the
        specified axis.

        Args:
            array (np.ndarray): Input array
            comparator (Callable): Comparison function to determine local extrema
            order (int): How many points on each side to use for the comparison

        Returns:
            np.ndarray: Boolean mask of local extrema (empty if array too small)
        """
        mask: np.ndarray = np.zeros_like(array, dtype=bool)
        # Return empty mask for arrays too small for extrema detection
        if array.shape[1] < 2 * order + 1:
            return mask
        idx = scipy.signal.argrelextrema(
            array, comparator, axis=1, order=order
        )
        mask[idx] = array[idx] > self.params.min_peak_amplitude
        return mask

    def _find_liquid_layers(
        self,
        beta: np.ndarray,
        beta_diff: np.ndarray,
        height: np.ndarray,
        win_below: int,
        win_above: int,
        min_height: float,
    ) -> np.ndarray:
        # Get the local maxima of the beta profile to be used as candidate peaks
        candidate_peaks: np.ndarray = self._find_local_extrema(
            beta, np.greater, 4
        )

        liquid_layers: np.ndarray = np.zeros_like(candidate_peaks, dtype=bool)

        for t, peak_idx in zip(*np.where(candidate_peaks)):
            lprof = beta[t, :]
            dprof = beta_diff[t, :]

            # Define search window around the peak
            start: int = max(0, peak_idx - win_below)
            end: int = min(peak_idx + win_above, lprof.shape[0])

            # Find base: lowermost point where gradient exceeds threshold
            # This follows CloudnetPy's ind_base logic
            diffs_below = dprof[start:peak_idx]
            if diffs_below.size == 0:
                continue
            base_threshold = (
                np.max(diffs_below) / self.params.peak_threshold_factor
            )
            valid_base_indices = np.where(diffs_below > base_threshold)
            if valid_base_indices[0].size == 0:
                continue
            base = start + valid_base_indices[0][0]

            # Find top: uppermost point where gradient is below threshold
            # This follows CloudnetPy's ind_top logic
            diffs_above = dprof[peak_idx:end]
            if diffs_above.size == 0:
                continue
            top_threshold = (
                np.min(diffs_above) / self.params.peak_threshold_factor
            )
            valid_top_indices = np.where(diffs_above < top_threshold)
            if valid_top_indices[0].size == 0:
                continue
            top = peak_idx + valid_top_indices[0][-1] + 1

            # Verify if the identified peak meets the criteria for a valid liquid layer
            if self._is_qualified_peak(
                beta[t, :], height, top, base, peak_idx, min_height
            ) and self._passes_peak_contrast(
                lprof,
                height,
                peak_idx,
            ):
                # Set the found layer as liquid (inclusive of top)
                liquid_layers[t, base : top + 1] = True

        return liquid_layers

    def transform(self, data: Dataset) -> DataArray:
        # Transform backscatter into liquid layers Tuononen et al. (2019) algorithm

        np_err_orig = np.geterr()
        np.seterr(all="warn")

        # Create copy of the beta so the original profile is not modified
        beta: DataArray = data["beta"].copy()

        # Ensure the dimensionality
        if beta.dims != ("time", "height"):
            raise ValueError(
                f"Input dimensions must be ('time', 'height'). Found dimensions: {beta.dims}"
            )

        # Calculate actual height spacing using median difference to match CloudnetPy's n_elements
        # This ensures correct window size calculations and robustness to non-uniform grids
        d_height: float = float(np.nanmedian(np.diff(beta.height)))

        # Compute the number of grid points for the search windows
        # Use round (not ceil) to match CloudnetPy's int(np.round(...))
        win_below: int = int(
            np.round(self.params.peak_search_win_below / d_height)
        )
        win_above: int = int(
            np.round(self.params.peak_search_win_above / d_height)
        )

        # Compute raw differences between subsequent gates
        beta_diff_values: np.ndarray = np.diff(beta.values, axis=1)
        # Diff is computed as value[i+1] - value[i], so we append a column of zeros at the end to maintain the same shape
        beta_diff_values = np.concatenate(
            [beta_diff_values, np.zeros((beta.shape[0], 1))], axis=1
        )

        # Convert back to DataArray for compatibility with xr.apply_ufunc
        beta_diff: DataArray = xr.DataArray(
            beta_diff_values,
            coords=beta.coords,
            dims=beta.dims,
            name="beta_diff",
        )

        beta = beta.fillna(0.0)
        beta_diff = beta_diff.fillna(0.0)

        func = partial(
            self._find_liquid_layers,
            win_below=win_below,
            win_above=win_above,
            min_height=float(beta.height[0]),
        )

        liquid_mask: DataArray = xr.apply_ufunc(
            func,
            beta,
            beta_diff,
            beta.height,
            output_dtypes=[bool],
        )

        liquid_mask = liquid_mask.rename("liquid_cloud_layers")

        np.seterr(**np_err_orig)

        return liquid_mask
