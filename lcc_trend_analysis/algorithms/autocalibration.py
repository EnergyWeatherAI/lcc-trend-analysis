import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from lcc_trend_analysis.logging import get_logger


logger = get_logger(__name__)


@dataclass
class LidarRatioComputationParameters:
    """Parameters controlling candidate stratocumulus profile detection."""

    max_peak_height: float = 2000.0
    min_peak_height: float = 1000.0

    min_peak_amplitude: float = 1e-4
    peak_contrast_factor: float = 20.0
    peak_contrast_distance: float = 300.0

    oversaturation_negative_layer_threshold: float = 100.0
    max_aerosol_beta_fraction: float = 0.05
    beta_integral_upper_bound: float = 2400.0

    consistency_window_size: int = 6
    consistency_threshold: float = 0.05
    consistency_min_neighbors: int = 3
    min_samples_per_day: int = 30
    nighttime_min_solar_zenith_angle: float = 96.0 # night excluding twilight


@dataclass
class AutocalibrationParameters:
    """Parameters for autocalibration."""

    min_samples_for_autocalibration: int = 3
    rolling_median_min_samples: int = 30

    multiple_scattering_factor: float = (
        0.78  # instrument & height dependent, but should be constant in time
    )


class Autocalibrator:
    """Calibrator class that calculates lidar ratio and scaling factor for autocalibration."""

    def __init__(
        self,
        lr_params: Optional[LidarRatioComputationParameters] = None,
        ac_params: Optional[AutocalibrationParameters] = None,
        calibration_breakpoints_path: Optional[Path] = None,
    ):
        """Initialize the calibrator with separate profile and factor settings."""
        if lr_params is None:
            lr_params = LidarRatioComputationParameters()
        if ac_params is None:
            ac_params = AutocalibrationParameters()
        self.lr_params = lr_params
        self.ac_params = ac_params
        self._calibration_breakpoints = self._load_calibration_breakpoints(
            calibration_breakpoints_path
        )

    @staticmethod
    def _normalize_breakpoint_key(value: str) -> str:
        """Normalize site and instrument keys for breakpoint lookups."""
        return str(value).strip().lower()

    @classmethod
    def _load_calibration_breakpoints(
        cls,
        calibration_breakpoints_path: Optional[Path],
    ) -> dict[tuple[str, str], pd.DatetimeIndex]:
        """Load manual calibration breakpoints keyed by site and instrument."""
        if calibration_breakpoints_path is None:
            return {}

        breakpoint_path = Path(calibration_breakpoints_path)
        if not breakpoint_path.exists():
            logger.error(
                "Calibration breakpoint file not found at %s",
                breakpoint_path,
            )
            raise FileNotFoundError(
                f"Calibration breakpoint file not found at {breakpoint_path}"
            )

        with breakpoint_path.open("r", encoding="utf-8") as f:
            raw_breakpoints = json.load(f)

        normalized_breakpoints: dict[tuple[str, str], pd.DatetimeIndex] = {}
        for site_id, site_breakpoints in raw_breakpoints.items():
            if not isinstance(site_breakpoints, dict):
                raise ValueError(
                    "Calibration breakpoints must map site IDs to instrument-date mappings"
                )

            site_key = cls._normalize_breakpoint_key(site_id)
            for instrument_name, breakpoint_dates in site_breakpoints.items():
                instrument_key = cls._normalize_breakpoint_key(instrument_name)
                breakpoint_index = pd.Index(pd.to_datetime(list(breakpoint_dates)))
                normalized_breakpoints[(site_key, instrument_key)] = pd.DatetimeIndex(
                    breakpoint_index.dropna().drop_duplicates().sort_values()
                )

        logger.debug(
            "Loaded calibration breakpoints from %s for %d site-instrument pairs",
            breakpoint_path,
            len(normalized_breakpoints),
        )
        for (site_key, instrument_key), breakpoint_index in normalized_breakpoints.items():
            logger.debug(
                "Calibration breakpoints for %s %s: %s",
                site_key,
                instrument_key,
                ", ".join(str(ts.date()) for ts in breakpoint_index)
                if len(breakpoint_index) > 0
                else "none",
            )

        return normalized_breakpoints

    def _get_calibration_breakpoints(
        self,
        site_id: str,
        instrument_name: str,
    ) -> pd.DatetimeIndex:
        """Return the manual breakpoint dates for one site-instrument pair."""
        return self._calibration_breakpoints.get(
            (
                self._normalize_breakpoint_key(site_id),
                self._normalize_breakpoint_key(instrument_name),
            ),
            pd.DatetimeIndex([]),
        )

    @staticmethod
    def _nantrapz(y, x, axis=-1):
        """Like np.trapz, but skips NaNs in y and x."""
        y = np.asarray(y)
        x = np.asarray(x)
        mask = np.isfinite(y) & np.isfinite(x)
        if not np.any(mask):
            return np.nan
        y = y[mask]
        x = x[mask]
        if y.size < 2:
            return np.nan
        return np.trapezoid(y, x, axis=axis)

    # Integrate beta_raw over range, profile by profile
    @staticmethod
    def _integrate_beta(da):
        """Integrate backscatter along range while preserving the profile axis.

        ``xr.apply_ufunc`` is used here so the same helper works for both eager
        and dask-backed datasets without manually branching on dimensionality.
        """
        return xr.apply_ufunc(
            Autocalibrator._nantrapz,
            da,
            da["range"],
            input_core_dims=[["range"], ["range"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    def compute_candidate_lidar_ratio(
        self,
        l1b_ds: xr.Dataset,
        solar_zenith_angle: Optional[xr.DataArray] = None,
    ) -> tuple[float, float]:
        """
        Computes the temporal median lidar ratio for a specific candidate stratocumulus period.
        O'Connor et al. (2004), Eq 6. LR = 1 / (2 * integral of beta(r) dr)
        Checks following Hopkin et al. (2019) criteria to ensure valid candidate.
        When solar zenith angle is provided, only nighttime profiles are kept.
        """
        null_return = (np.nan, np.nan)

        if "beta_raw" not in l1b_ds:
            return null_return

        # Ensure range is a coordinate so it won't get dropped when selecting beta_raw
        l1b_ds = l1b_ds.swap_dims({"height": "range"})
        beta_raw = l1b_ds["beta_raw"]

        # Mask out daylight-contamined profiles
        if solar_zenith_angle is not None:
            if "time" not in solar_zenith_angle.dims:
                raise ValueError(
                    "solar_zenith_angle must have a 'time' dimension"
                )

            solar_zenith_angle = solar_zenith_angle.reindex(
                time=beta_raw["time"],
                copy=False,
            )
            nighttime_profiles = np.isfinite(solar_zenith_angle) & (
                solar_zenith_angle
                > self.lr_params.nighttime_min_solar_zenith_angle
            )
            
            beta_raw = beta_raw.where(nighttime_profiles)

            if not beta_raw.notnull().any():
                return null_return

        dr = beta_raw["range"].diff(dim="range").median().item()

        # 1) Select profiles with strong return in desired range (2-4 km for CHM15k, 1-2.4 km for others)
        candidate_peaks = beta_raw.where(
            (beta_raw["range"] >= self.lr_params.min_peak_height)
            & (beta_raw["range"] <= self.lr_params.max_peak_height)
            & (beta_raw >= self.lr_params.min_peak_amplitude)
        )
        # Mask profiles with no valid peaks in the desired range
        valid_profiles = candidate_peaks.notnull().any(dim="range")
        beta_raw = beta_raw.where(valid_profiles)

        if not beta_raw.notnull().any():
            return null_return

        # 2) Reject profiles where peak is not at least 20 stronger than values 300 m above or below.
        # Get the range of the maximum beta in the candidate range
        idx = candidate_peaks.fillna(-np.inf).argmax(dim="range")
        peak_value = candidate_peaks.max(dim="range")
        range_of_max_beta = beta_raw["range"].isel(range=idx)

        peak_value = peak_value.where(valid_profiles)
        range_of_max_beta = range_of_max_beta.where(valid_profiles)

        # Values 300 m above and below the peak
        above_range = range_of_max_beta + self.lr_params.peak_contrast_distance
        below_range = range_of_max_beta - self.lr_params.peak_contrast_distance

        above_value = beta_raw.sel(range=above_range, method="nearest")
        below_value = beta_raw.sel(range=below_range, method="nearest")

        # Check the contrast condition for each profile
        contrast_ok = (
            peak_value >= self.lr_params.peak_contrast_factor * above_value
        ) & (peak_value >= self.lr_params.peak_contrast_factor * below_value)

        # Mask entire profiles that fail the contrast check
        beta_raw = beta_raw.where(contrast_ok)

        if not beta_raw.notnull().any():
            return null_return

        # 3) Reject profiles with more than 100 m thick continuous negative-beta layer (indicative of oversaturation).
        negative_mask = beta_raw.notnull() & (beta_raw < 0)
        negative_run_lengths = negative_mask.rolling(
            {
                "range": int(
                    np.ceil(
                        self.lr_params.oversaturation_negative_layer_threshold
                        / dr
                    )
                )
            },
            min_periods=int(
                np.ceil(
                    self.lr_params.oversaturation_negative_layer_threshold / dr
                )
            ),
        ).sum()
        has_oversaturated_layer = (
            negative_run_lengths
            >= self.lr_params.oversaturation_negative_layer_threshold / dr
        ).any(dim="range")

        beta_raw = beta_raw.where(~has_oversaturated_layer)

        if not beta_raw.notnull().any():
            return null_return

        # 4) Compute integrated backscatter, integrate only to a given upper bound (4 km for CHM15k, 2.4 km for others)
        beta_raw = beta_raw.where(
            beta_raw["range"]
            <= above_range.where(
                above_range <= self.lr_params.beta_integral_upper_bound,
                other=self.lr_params.beta_integral_upper_bound,
            ),
        )

        bint = Autocalibrator._integrate_beta(beta_raw)

        # 5) Filter aerosols by ensuring the integral of beta under the cloud is < 0.05 of total profile integral
        bint_aerosol = Autocalibrator._integrate_beta(
            beta_raw.where(beta_raw["range"] < below_range)
        )

        bint = bint.where(
            (bint_aerosol / bint) < self.lr_params.max_aerosol_beta_fraction,
        )

        if not bint.notnull().any():
            return null_return

        # 6) Compute the lidar ratio for remaining profiles
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lidar_ratios = 1.0 / (2.0 * bint)

        # 7) Consistency check: three preceding and following samples in time must be within 5% of median of its neighbors
        window = self.lr_params.consistency_window_size
        center = self.lr_params.consistency_window_size // 2

        # Rolling window along 'time' (or 'profile' if that's the dimension)
        rolling = lidar_ratios.rolling(
            {"time": window}, center=True, min_periods=center + 1
        )

        def check_consistency(arr, **kwargs):
            if not hasattr(arr, "__len__") or len(arr) <= center:
                # len-1 scalar from reduction
                return np.nan
            center_val = arr[center]
            if not np.isfinite(center_val):
                return np.nan
            # Exclude center and NaNs
            neighbors = np.delete(arr, center)
            neighbors = neighbors[np.isfinite(neighbors)]
            if neighbors.size < self.lr_params.consistency_min_neighbors:
                return np.nan
            within_threshold = np.abs(center_val - np.median(neighbors)) <= (
                self.lr_params.consistency_threshold * np.median(neighbors)
            )
            return center_val if within_threshold else np.nan

        lidar_ratios = xr.apply_ufunc(
            check_consistency,
            rolling.construct("window_dim"),
            input_core_dims=[["window_dim"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        """
        lidar_ratios = lidar_ratios.where(
           np.abs(lidar_ratios - lidar_ratios.median())
           <= self.lr_params.consistency_threshold
        )
        """

        # Ensure at least n valid values per day
        if (
            lidar_ratios.notnull().count().item()
            < self.lr_params.min_samples_per_day
        ):
            return null_return

        # 8) Return daily median of remaining lidar ratios as candidate lidar ratio for this profile
        # + the range of max beta (CBH) as a sanity check (not used in autocalibration factor computation)
        return float(lidar_ratios.median().item()), float(
            range_of_max_beta.where(lidar_ratios.notnull()).median().item()
        )

    def get_autocalibration_factor(
        self,
        candidates_df: pd.DataFrame,
        site_id: str,
        instrument_name: str,
        target_date,
        instrument_sc_lidar_ratio: float,
    ) -> float:
        """Convenience wrapper that filters a global candidate table to one group.

        Most of the real work happens in the group-level bulk implementation.
        This helper exists for callers that only need a single site, instrument,
        and date without manually pre-grouping the input table.
        """
        if candidates_df is None or candidates_df.empty:
            return 1.0

        df = candidates_df[
            (candidates_df["site_id"] == site_id)
            & (candidates_df["instrument_name"] == instrument_name)
        ]
        return self.get_autocalibration_factor_from_group(
            df,
            site_id,
            instrument_name,
            target_date,
            instrument_sc_lidar_ratio,
        )

    def get_autocalibration_factor_from_group(
        self,
        candidate_group_df: pd.DataFrame,
        site_id: str,
        instrument_name: str,
        target_date,
        instrument_sc_lidar_ratio: float,
    ) -> float:
        """Compute one factor from an already filtered site-instrument group.

        This is a thin single-date adapter around the bulk factor builder so the
        segment-aware rolling-median logic stays defined in exactly one place.
        Single-date callers still honor the configured manual calibration
        breakpoints for the requested site and instrument.
        """
        factors_df = self.get_autocalibration_factors_from_group(
            candidate_group_df,
            site_id,
            instrument_name,
            [target_date],
            instrument_sc_lidar_ratio,
        )
        if factors_df.empty:
            return 1.0

        median_at_target = factors_df.iloc[0]["autocalibration_factor"]

        if not np.isfinite(median_at_target):
            return 1.0

        return float(median_at_target)

    @staticmethod
    def _empty_factor_frame() -> pd.DataFrame:
        """Return the stable empty output schema for factor computation.

        Keeping the schema centralized matters because downstream parquet merges
        and joins rely on these columns existing even when no factors are found.
        """
        return pd.DataFrame(
            columns=[
                "date",
                "autocalibration_factor",
                "factor_status",
                "candidate_lidar_ratio",
                "rolling_median_lidar_ratio",
                "instrument_sc_lidar_ratio",
                "n_valid_candidate_lidar_ratios",
                "scaling",
                "background_noise_p10",
                "background_noise_p25",
                "background_noise_p75",
                "background_noise_p90",
            ]
        )

    @staticmethod
    def _prepare_target_dates(target_dates) -> pd.DataFrame:
        """Normalize requested dates into a unique sorted index for factor lookup.

        Keeping this canonicalized in one place makes the downstream merges and
        interpolation rules deterministic even if callers provide duplicates,
        strings, or unsorted timestamps.
        """
        target_index = pd.Index(pd.to_datetime(list(target_dates)))
        target_index = target_index.dropna().drop_duplicates().sort_values()
        if target_index.empty:
            return pd.DataFrame(columns=["date"])
        return pd.DataFrame({"date": target_index})

    @staticmethod
    def _prepare_candidate_group(
        candidate_group_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a clean daily candidate series for one site-instrument group.

        The factor logic assumes one positive lidar-ratio candidate per day.
        This helper enforces that contract so the autocalibration code can work
        with a monotonic daily time series instead of repeatedly handling input
        irregularities.
        """
        if candidate_group_df is None or candidate_group_df.empty:
            return pd.DataFrame(columns=["date_ts", "lidar_ratio"])

        if "lidar_ratio" not in candidate_group_df.columns:
            return pd.DataFrame(columns=["date_ts", "lidar_ratio"])

        df = candidate_group_df.copy()
        if "date_ts" in df.columns:
            df["date_ts"] = pd.to_datetime(df["date_ts"])
        elif "date" in df.columns:
            df["date_ts"] = pd.to_datetime(df["date"])
        else:
            return pd.DataFrame(columns=["date_ts", "lidar_ratio"])

        df = df.dropna(subset=["date_ts", "lidar_ratio"])
        df = df[df["lidar_ratio"] > 0]
        if df.empty:
            return pd.DataFrame(columns=["date_ts", "lidar_ratio"])

        return df.sort_values(by="date_ts", ascending=True).drop_duplicates(
            subset=["date_ts"], keep="last"
        )

    def _assign_calibration_segments(
        self,
        date_index: pd.DatetimeIndex,
        site_id: str,
        instrument_name: str,
    ) -> np.ndarray:
        """Assign dates to manual calibration segments for one instrument.

        Each configured breakpoint date is treated as the first day of a new
        calibration segment. Dates for site-instrument pairs without configured
        breakpoints remain in a single uninterrupted segment.
        """
        if len(date_index) == 0:
            return np.zeros(0, dtype=int)

        breakpoints = self._get_calibration_breakpoints(site_id, instrument_name)
        if breakpoints.empty:
            return np.zeros(len(date_index), dtype=int)

        return np.searchsorted(
            breakpoints.to_numpy(),
            pd.DatetimeIndex(date_index).to_numpy(),
            side="right",
        ).astype(int)

    def _resolve_factors_from_series(
        self,
        candidate_series: pd.Series,
        target_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Resolve rolling-median factors from one calibration segment.

        The factor series is built from a centered rolling quantile estimate of
        the valid candidate lidar ratios. A half-year window is attempted
        first, then any still-missing dates are retried with one-year and
        two-year windows before time interpolation and nearest-valid fallback
        are used. The caller is responsible for passing only one calibration
        segment so these steps never cross a segment boundary.
        """
        segment_result = pd.DataFrame(index=target_index)
        segment_result["rolling_median_lidar_ratio"] = np.nan
        segment_result["factor_status"] = "default_no_candidates"

        if candidate_series.empty:
            return segment_result

        candidate_median = float(candidate_series.median())

        min_candidates = max(
            1, int(self.ac_params.min_samples_for_autocalibration)
        )
        if len(candidate_series) < min_candidates:
            if np.isfinite(candidate_median):
                segment_result["rolling_median_lidar_ratio"] = candidate_median
                segment_result["factor_status"] = "candidate_median_fallback"
            else:
                segment_result["factor_status"] = "default_insufficient_candidates"
            return segment_result

        rolling_median = pd.Series(np.nan, index=candidate_series.index)
        rolling_status = np.full(len(candidate_series), None, dtype=object)
        rolling_windows = [
            ("rolling_two_year", pd.Timedelta(days=2.0 * 365.25)),
            ("rolling_three_year", pd.Timedelta(days=3.0 * 365.25)),
            ("rolling_four_year", pd.Timedelta(days=4.0 * 365.25))
        ]

        for status_label, window in rolling_windows:
            window_median = candidate_series.rolling(
                window=window,
                min_periods=self.ac_params.rolling_median_min_samples,
                center=True,
            ).quantile(0.4)
            fill_mask = (~np.isfinite(rolling_median.to_numpy())) & np.isfinite(
                window_median.to_numpy()
            )
            if fill_mask.any():
                rolling_median.iloc[fill_mask] = window_median.iloc[fill_mask]
                rolling_status[fill_mask] = status_label

        valid_medians = rolling_median[np.isfinite(rolling_median)]
        if valid_medians.empty:
            if np.isfinite(candidate_median):
                segment_result["rolling_median_lidar_ratio"] = candidate_median
                segment_result["factor_status"] = "candidate_median_fallback"
            else:
                segment_result["factor_status"] = "default_no_valid_medians"
            return segment_result

        raw_target_medians = rolling_median.reindex(target_index)
        raw_target_status = pd.Series(
            rolling_status,
            index=candidate_series.index,
            dtype=object,
        ).reindex(target_index)
        # Fill target dates between valid rolling-median estimates without
        # changing the underlying windowed median calculation itself.
        rolling_union = rolling_median.reindex(
            rolling_median.index.union(target_index)
        ).sort_index()
        interpolated_medians = rolling_union.interpolate(
            method="time",
            limit_direction="both",
        ).reindex(target_index)

        resolved_medians = interpolated_medians.copy()
        factor_status = pd.Series(
            np.where(
                np.isfinite(raw_target_medians.to_numpy()),
                raw_target_status.to_numpy(),
                "interpolated",
            ),
            index=target_index,
            dtype=object,
        )

        missing_mask = ~np.isfinite(resolved_medians.to_numpy())
        if missing_mask.any():
            nearest_positions = valid_medians.index.get_indexer(
                target_index[missing_mask],
                method="nearest",
            )
            nearest_values = valid_medians.to_numpy()[nearest_positions]
            resolved_medians.iloc[missing_mask] = nearest_values
            factor_status.loc[target_index[missing_mask]] = "nearest_valid"

        segment_result["rolling_median_lidar_ratio"] = (
            resolved_medians.to_numpy()
        )
        segment_result["factor_status"] = factor_status.to_numpy()
        return segment_result

    def get_autocalibration_factors_from_group(
        self,
        candidate_group_df: pd.DataFrame,
        site_id: str,
        instrument_name: str,
        target_dates,
        instrument_sc_lidar_ratio: float,
    ) -> pd.DataFrame:
        """Compute autocalibration factors for one site-instrument time series.

        The workflow is: normalize requested dates, clean the candidate series,
        split both candidate and target dates according to the configured manual
        calibration breakpoints, and then resolve a centered yearly running
        median independently within each segment before mapping the resulting
        factor estimates onto the requested dates.
        """
        prepared_target_dates = self._prepare_target_dates(target_dates)
        if prepared_target_dates.empty:
            return self._empty_factor_frame()

        factors_df = prepared_target_dates.set_index("date")
        target_index = pd.DatetimeIndex(factors_df.index)

        factors_df["candidate_lidar_ratio"] = np.nan
        factors_df["rolling_median_lidar_ratio"] = np.nan
        factors_df["instrument_sc_lidar_ratio"] = instrument_sc_lidar_ratio
        factors_df["n_valid_candidate_lidar_ratios"] = 0
        factors_df["autocalibration_factor"] = 1.0
        factors_df["factor_status"] = "default_no_candidates"

        if instrument_sc_lidar_ratio <= 0:
            factors_df["factor_status"] = "default_invalid_instrument_ratio"
            return factors_df.reset_index()

        candidate_df = self._prepare_candidate_group(candidate_group_df)
        if candidate_df.empty:
            factors_df["factor_status"] = "default_no_candidates"
            return factors_df.reset_index()

        candidate_series = candidate_df.set_index("date_ts")[
            "lidar_ratio"
        ].sort_index()
        factors_df["candidate_lidar_ratio"] = factors_df.index.to_series().map(
            candidate_series
        )

        factors_df["n_valid_candidate_lidar_ratios"] = len(candidate_df)

        breakpoint_index = self._get_calibration_breakpoints(site_id, instrument_name)
        if breakpoint_index.empty:
            logger.debug(
                "No manual calibration breakpoints configured for %s %s; using one segment for %d target dates",
                site_id,
                instrument_name,
                len(target_index),
            )
        else:
            logger.debug(
                "Applying %d manual calibration breakpoints to %s %s across %d target dates and %d candidate lidar-ratio dates",
                len(breakpoint_index),
                site_id,
                instrument_name,
                len(target_index),
                len(candidate_series),
            )

        target_segment_ids = self._assign_calibration_segments(
            target_index,
            site_id,
            instrument_name,
        )
        candidate_segment_ids = self._assign_calibration_segments(
            pd.DatetimeIndex(candidate_series.index),
            site_id,
            instrument_name,
        )
        for segment_id in np.unique(target_segment_ids):
            target_mask = target_segment_ids == segment_id
            segment_targets = target_index[target_mask]
            segment_candidates = candidate_series[
                candidate_segment_ids == segment_id
            ]
            logger.debug(
                "Calibration segment %d for %s %s spans %s to %s with %d target dates and %d candidate lidar-ratio dates",
                int(segment_id),
                site_id,
                instrument_name,
                segment_targets.min().date(),
                segment_targets.max().date(),
                len(segment_targets),
                len(segment_candidates),
            )
            segment_result = self._resolve_factors_from_series(
                segment_candidates,
                segment_targets,
            )
            factors_df.loc[
                target_mask,
                "rolling_median_lidar_ratio",
            ] = segment_result["rolling_median_lidar_ratio"].to_numpy()
            factors_df.loc[target_mask, "factor_status"] = segment_result[
                "factor_status"
            ].to_numpy()

        finite_factor_mask = np.isfinite(
            factors_df["rolling_median_lidar_ratio"].to_numpy()
        )
        factors_df.loc[finite_factor_mask, "autocalibration_factor"] = (
            factors_df.loc[finite_factor_mask, "rolling_median_lidar_ratio"]
            / instrument_sc_lidar_ratio
            * self.ac_params.multiple_scattering_factor
        )

        return factors_df.reset_index()
