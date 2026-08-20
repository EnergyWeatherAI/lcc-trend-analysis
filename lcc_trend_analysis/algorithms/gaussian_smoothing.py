from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from ..type_aliases import DataArray, Dataset
from ..algorithms.range_correction import CeilometerRangeCorrectionTransformer

@dataclass
class GaussianSmoothingParameters:
    """Parameters for BackscatterGaussianSmoothingTransformer.
    Defaults are CloudnetPy defaults.

    Attributes:
        noise_top_gate_fraction (float): Fraction of top gates used for noise estimation.
        noise_min (float): Minimum noise level to avoid unrealistically low background noise estimates.
        noise_clean (float): Noise level to set for cleaned profiles.
        negative_filter_min_negative_gates (int): Minimum number of consecutive negative gates to trigger cleaning.
        negative_filter_skip_lowest_n_gates (int): Number of lowest gates to skip in negative filtering.
    """

    strong_cloud_threshold: float = 1e-6
    
    sigma_metres: float = 10.0
    sigma_seconds: float = 60.0


class BackscatterGaussianSmoothingTransformer:
    """Stateless transformer to smooth raw lidar backscatter data.

    This transformer applies a Gaussian smoothing to the raw backscatter profiles."""

    def __init__(
        self,
        smoothing_params: Optional[GaussianSmoothingParameters] = None,
    ):
        """Initialize the transformer with filtering parameters.

        Args:
            smoothing_params (GaussianSmoothingParameters | None): Parameters for smoothing
        """
        if smoothing_params is None:
            smoothing_params = GaussianSmoothingParameters()
        self.params: GaussianSmoothingParameters = smoothing_params

    def smooth(
        self,
        data: Dataset,
    ) -> Dataset:
        """Applies Gaussian smoothing to the raw backscatter profiles in the input dataset.

        Args:
            data (Dataset): Input dataset containing raw backscatter data with key "beta_raw"

        Returns:
            Dataset: Dataset with smoothed backscatter profiles added as "beta_smooth"
        """
        # Apply filtering filters in sequence following CloudnetPy approach

        # Create a copy of the raw beta data to avoid modifying the original
        assert "beta_raw" in data.data_vars, (
            "Input data must contain 'beta_raw' variable."
        )
        assert "range" in data.coords, (
            "Input data must contain 'range' coordinate for filtering."
        )

        smoothed_beta: DataArray = data["beta_raw"].copy()
        
        range_correction_transformer = CeilometerRangeCorrectionTransformer()
        smoothed_beta = range_correction_transformer.inverse_transform(
            smoothed_beta, range_var=data["range"]
        )
        
        # First we need to extract strong clouds from beta
        strong_cloud_mask = smoothed_beta > self.params.strong_cloud_threshold
        beta_strong_clouds = smoothed_beta.where(strong_cloud_mask)
        
        # Before smoothing limit the strong cloud values to the threshold to avoid smearing strong clouds into noise
        smoothed_beta = smoothed_beta.where(~strong_cloud_mask, other=self.params.strong_cloud_threshold)
        
        # Compute sigma for Gaussian smoothing on range and time dimensions
        tdiff = np.median(np.diff(data["time"].values)) / np.timedelta64(1, 's')  # Time resolution in seconds
        rdiff = np.median(np.diff(data["range"].values))  # Range resolution
        
        sigma_time = self.params.sigma_seconds / tdiff
        sigma_range = self.params.sigma_metres / rdiff
        
        # Apply Gaussian smoothing using gaussian_filter from scipy.ndimage
        smoothed_beta_values = gaussian_filter(smoothed_beta.values, sigma=[sigma_time, sigma_range])
        smoothed_beta[:] = smoothed_beta_values
        
        # Restore the strong cloud values after smoothing to avoid smearing them into noise
        smoothed_beta = smoothed_beta.where(~strong_cloud_mask, other=beta_strong_clouds)
        
        smoothed_beta = range_correction_transformer.transform(
            smoothed_beta, range_var=data["range"]
        )
        
        data["beta_smooth"] = smoothed_beta

        return data
