from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from numpy._typing._array_like import NDArray
from scipy import stats as spstats
from sklearn.base import BaseEstimator, _fit_context  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]
from statsmodels.tsa import stattools  # type: ignore[import-untyped]


class MannKendallTrendEstimator(BaseEstimator):
    """Hirsch-Slack seasonal Mann-Kendall/Theil-Sen trend estimator with 3PW prewhitening.

    For prewhitening, the default ``3pw`` mode follows the same algorithmic structure as the
    ``mannkendall`` package, but accepts ``numpy.datetime64`` input directly.

    Parameters
    ----------
    resolution : float, default=1e-6
            Measurement resolution used when counting value ties.
    pw_method : {"pw", "tfpw_y", "tfpw_ws", "vctfpw", "3pw"}, default="3pw"
            Prewhitening method used for the trend estimate.
    alpha_mk : float, default=95.0
            Significance level used for the Mann-Kendall test decision.
    alpha_cl : float, default=90.0
            Confidence level used for Sen slope confidence intervals.
    alpha_ak : float, default=95.0
            Significance level used when testing lag-1 autocorrelation.
    same_instrument_only : bool, default=False
            Restrict pairwise trend estimation and MK comparisons to
            within-instrument data.
    same_season_only : bool, default=False
            Restrict pairwise trend estimation and MK comparisons to observations
            from the same meteorological season (DJF/MAM/JJA/SON).
    n_jobs : int, default=-1
            Number of parallel workers used for pairwise slope computation.
    """

    _parameter_constraints = {
        "resolution": [float, int],
        "pw_method": [str],
        "alpha_mk": [float, int],
        "alpha_cl": [float, int],
        "alpha_ak": [float, int],
        "same_instrument_only": [bool],
        "same_season_only": [bool],
        "n_jobs": [int],
    }

    VALID_PW_METHODS = {"pw", "tfpw_y", "tfpw_ws", "vctfpw", "3pw"}
    SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.25
    MIN_VALID_SAMPLES = 10
    MIN_PREWHITEN_DENOM = 1e-3

    def __init__(
        self,
        resolution: float = 1e-6,
        pw_method: str = "3pw",
        alpha_mk: float = 95.0,
        alpha_cl: float = 90.0,
        alpha_ak: float = 95.0,
        same_instrument_only: bool = False,
        same_season_only: bool = False,
        n_jobs: int = -1,
    ):
        self.resolution = resolution
        self.pw_method = pw_method
        self.alpha_mk = alpha_mk
        self.alpha_cl = alpha_cl
        self.alpha_ak = alpha_ak
        self.same_instrument_only = same_instrument_only
        self.same_season_only = same_season_only
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float_array(values: Any) -> np.ndarray:
        """Return a 1-D float64 observation array."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError("Expected a 1-D array of observations.")
        return array

    @staticmethod
    def _as_datetime64_array(times: Any) -> np.ndarray:
        """Return a 1-D datetime64[ns] time array."""
        array = np.asarray(times)
        if array.ndim != 1:
            raise ValueError("Expected a 1-D array of observation times.")
        try:
            return array.astype("datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Observation times must be convertible to numpy datetime64."
            ) from exc

    @staticmethod
    def _as_instrument_array(
        instrument_labels: Any | None, n_obs: int
    ) -> np.ndarray:
        """Return normalized 1-D instrument labels aligned with observations."""
        if instrument_labels is None:
            return np.full(n_obs, "__all__", dtype=object)

        array = np.asarray(instrument_labels, dtype=object)
        if array.ndim != 1:
            raise ValueError("Expected a 1-D array of instrument labels.")
        if array.size != n_obs:
            raise ValueError(
                "Instrument labels must have the same length as the observations."
            )

        normalized = np.empty(n_obs, dtype=object)
        for index, label in enumerate(array):
            if label is None or (isinstance(label, float) and np.isnan(label)):
                normalized[index] = "__missing__"
            else:
                normalized[index] = str(label)
        return normalized

    @staticmethod
    def _timedelta64_to_seconds(delta: np.ndarray) -> np.ndarray:
        """Convert timedeltas to floating-point seconds."""
        return delta.astype("timedelta64[ns]").astype(np.float64) / 1e9

    @staticmethod
    def _timestamp_tie_counts(
        obs_dts: np.ndarray, obs: np.ndarray
    ) -> np.ndarray:
        """Count valid observations that share the same timestamp."""
        valid_obs_dts = obs_dts[~np.isnan(obs)]
        if valid_obs_dts.size == 0:
            return np.array([np.nan], dtype=np.float64)
        _, tie_counts = np.unique(valid_obs_dts, return_counts=True)
        return tie_counts.astype(np.float64)

    @staticmethod
    def _month_numbers(obs_dts: np.ndarray) -> np.ndarray:
        """Return calendar month numbers in [1, 12] for each timestamp."""
        month_index = obs_dts.astype("datetime64[M]").astype(np.int64)
        return month_index % 12 + 1

    def _season_labels(self, obs_dts: np.ndarray) -> np.ndarray:
        """Return meteorological season labels for each timestamp."""
        months = self._month_numbers(obs_dts)
        seasons = np.empty(obs_dts.shape, dtype=object)
        is_djf = (months == 12) | (months <= 2)
        is_mam = (months >= 3) & (months <= 5)
        is_jja = (months >= 6) & (months <= 8)
        seasons[is_djf] = "DJF"
        seasons[is_mam] = "MAM"
        seasons[is_jja] = "JJA"
        seasons[~(is_djf | is_mam | is_jja)] = "SON"
        return seasons

    def _build_strata_labels(
        self,
        obs_dts: np.ndarray,
        instrument_labels: np.ndarray,
    ) -> np.ndarray | None:
        """Build per-observation strata labels for MK/Sen calculations."""
        use_instrument = bool(self.same_instrument_only)
        use_season = bool(self.same_season_only)
        if not (use_instrument or use_season):
            return None

        if use_season:
            season_labels = self._season_labels(obs_dts)
        else:
            season_labels = np.full(
                obs_dts.shape, "__all_seasons__", dtype=object
            )

        if use_instrument:
            instrument_part = instrument_labels
        else:
            instrument_part = np.full(
                obs_dts.shape, "__all_instruments__", dtype=object
            )

        if use_instrument and use_season:
            return np.array(
                [
                    f"{instrument}__{season}"
                    for instrument, season in zip(
                        instrument_part, season_labels
                    )
                ],
                dtype=object,
            )
        if use_instrument:
            return np.array(instrument_part, dtype=object)
        return np.array(season_labels, dtype=object)

    # ------------------------------------------------------------------
    # Core Mann-Kendall helpers
    # ------------------------------------------------------------------
    def _quantize(self, data: np.ndarray) -> np.ndarray:
        """Map observations to the resolution grid used for tie handling."""
        valid = ~np.isnan(data)
        result = np.full_like(data, np.nan, dtype=np.float64)
        result[valid] = np.rint(data[valid] / self.resolution)
        return result
    
    
    def _nb_tie(self, data: np.ndarray) -> np.ndarray:
        """Return sizes of tie groups at the configured resolution."""
        valid_data = data[~np.isnan(data)]

        if valid_data.size < 2:
            return np.array([np.nan], dtype=np.float64)

        quantized = self._quantize(valid_data)
        _, tie_counts = np.unique(quantized, return_counts=True)
        tie_counts = tie_counts[tie_counts > 1]

        if tie_counts.size == 0:
            return np.array([0.0], dtype=np.float64)

        return tie_counts.astype(np.float64)

    def _has_stable_prewhitening_denom(self, autocorr: float) -> bool:
        """Return whether dividing by ``1 - autocorr`` is numerically safe."""
        return bool(
            np.isfinite(autocorr) and autocorr < 1.0 - self.MIN_PREWHITEN_DENOM
        )

    @staticmethod
    def _kendall_var(
        data: np.ndarray,
        ties: np.ndarray,
        time_tie_counts: np.ndarray,
    ) -> float:
        """Compute Kendall variance with value ties and exact timestamp ties."""
        n_real = np.count_nonzero(~np.isnan(data))
        if n_real < 2:
            return np.nan
        var_s = (
            n_real * (n_real - 1) * (2 * n_real + 5)
            - np.nansum(ties * (ties - 1) * (2 * ties + 5))
            - np.nansum(
                time_tie_counts
                * (time_tie_counts - 1)
                * (2 * time_tie_counts + 5)
            )
        ) / 18.0
        if n_real > 2:
            var_s += (
                np.nansum(ties * (ties - 1) * (ties - 2))
                * np.nansum(
                    time_tie_counts
                    * (time_tie_counts - 1)
                    * (time_tie_counts - 2)
                )
                / (9.0 * n_real * (n_real - 1) * (n_real - 2))
            )
        var_s += (
            np.nansum(ties * (ties - 1))
            * np.nansum(time_tie_counts * (time_tie_counts - 1))
            / (2.0 * n_real * (n_real - 1))
        )
        return float(var_s)

    def _use_stratification(
        self,
        obs: np.ndarray,
        strata_labels: np.ndarray | None,
    ) -> bool:
        """Return whether stratum-based estimation should be used."""
        if strata_labels is None:
            return False
        valid_strata = strata_labels[~np.isnan(obs)]
        return np.unique(valid_strata).size > 1

    @staticmethod
    def _use_instrument_prewhitening(
        obs: np.ndarray,
        instrument_labels: np.ndarray,
    ) -> bool:
        """Return whether prewhitening should be applied per instrument."""
        valid_instruments = instrument_labels[~np.isnan(obs)]
        return np.unique(valid_instruments).size > 1

    def _compute_stratified_stats(
        self,
        obs: np.ndarray,
        obs_dts: np.ndarray,
        strata_labels: np.ndarray,
    ) -> tuple[int, float]:
        """Compute S and variance as sums of within-stratum contributions."""
        total_s = 0
        total_variance = 0.0

        valid_strata = strata_labels[~np.isnan(obs)]
        for stratum in np.unique(valid_strata):
            stratum_mask = strata_labels == stratum
            obs_stratum = obs[stratum_mask]
            obs_dts_stratum = obs_dts[stratum_mask]
            if np.count_nonzero(~np.isnan(obs_stratum)) < 2:
                continue

            ties = self._nb_tie(obs_stratum)
            s_value, time_tie_counts = self._s_test(
                obs_stratum, obs_dts_stratum
            )
            variance = self._kendall_var(obs_stratum, ties, time_tie_counts)
            total_s += s_value
            if np.isfinite(variance):
                total_variance += variance

        return total_s, float(total_variance)

    @staticmethod
    def _std_normal_var(s_value: int, var_s: float) -> float:
        """Convert the S statistic to its normal-score approximation."""
        if s_value == 0 or not np.isfinite(var_s) or var_s <= 0:
            return 0.0
        return float((s_value - np.sign(s_value)) / np.sqrt(var_s))

    def _pairwise_valid_mask(
        self,
        obs_dts_valid: np.ndarray,
        index: int,
        strata_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return admissible pair mask for one leading observation."""
        delta_seconds = self._timedelta64_to_seconds(
            obs_dts_valid[index + 1 :] - obs_dts_valid[index]
        )
        valid_delta = delta_seconds > 0.0
        if strata_labels is not None:
            valid_delta &= strata_labels[index + 1 :] == strata_labels[index]
        return valid_delta

    def _s_test(
        self, obs: np.ndarray, obs_dts: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """Compute the S statistic and counts of exact timestamp ties."""
        time_tie_counts = self._timestamp_tie_counts(obs_dts, obs)
        valid_mask = ~np.isnan(obs)
        obs_valid = obs[valid_mask]
        obs_dts_valid = obs_dts[valid_mask]

        if obs_valid.size < 2:
            return 0, time_tie_counts

        s_value = 0
        for index in range(obs_valid.size - 1):
            valid_delta = self._pairwise_valid_mask(obs_dts_valid, index)
            if np.any(valid_delta):
                q = self._quantize(obs_valid)
                delta_q = q[index + 1:][valid_delta] - q[index]
                s_value += int(np.sum(np.sign(delta_q)))

        return s_value, time_tie_counts

    def _sen_slope(
        self,
        obs_dts: np.ndarray,
        obs: np.ndarray,
        kendall_variance: float,
        alpha_cl: float,
        strata_labels: np.ndarray | None = None,
        cache_key: str | None = None,
    ) -> tuple[float, float, float]:
        """Compute Sen's slope and MK-inverted order-statistic confidence bounds.

        When ``cache_key`` is provided, the sorted pairwise-slope sample computed
        here is stored in ``self._pairwise_slope_cache_`` (populated in ``fit()``)
        and reused on subsequent calls with the same key, avoiding recomputation
        of the O(n^2) pairwise-slope generation when only ``alpha_cl`` changes.
        The cached sample also backs ``get_pairwise_slope_summary()``.
        """
        cached_entry = None
        if cache_key is not None:
            cached_entry = self._pairwise_slope_cache_.get(cache_key)

        if cached_entry is not None:
            pairwise_slopes = cached_entry["sorted_slopes"]
            n_slopes = cached_entry["n_slopes"]
            slope = cached_entry["slope"]
        else:
            valid_mask = ~np.isnan(obs)
            obs_dts_valid = obs_dts[valid_mask]
            obs_valid = obs[valid_mask]
            strata_labels_valid = None
            if strata_labels is not None:
                strata_labels_valid = strata_labels[valid_mask]

            if obs_valid.size < 2:
                return np.nan, np.nan, np.nan

            # Sen's slope is the median of all pairwise slopes. This is the costly
            # part of the estimator, so compute it in chunks that can be parallelized.
            slopes = self._compute_pairwise_slopes(
                obs_dts_valid,
                obs_valid,
                strata_labels=strata_labels_valid,
            )

            if not slopes:
                return np.nan, np.nan, np.nan

            pairwise_slopes = np.concatenate(slopes)
            pairwise_slopes.sort()

            n_slopes = pairwise_slopes.size
            if n_slopes % 2 == 1:
                slope = pairwise_slopes[(n_slopes - 1) // 2]
            else:
                slope = (
                    pairwise_slopes[n_slopes // 2 - 1]
                    + pairwise_slopes[n_slopes // 2]
                ) / 2.0

            if cache_key is not None:
                self._pairwise_slope_cache_[cache_key] = {
                    "sorted_slopes": pairwise_slopes,
                    "n_slopes": n_slopes,
                    "slope": float(slope),
                    "kendall_variance": (
                        float(kendall_variance)
                        if np.isfinite(kendall_variance)
                        else np.nan
                    ),
                }

        if not np.isfinite(kendall_variance) or kendall_variance < 0.0:
            return float(slope), np.nan, np.nan

        # Standard Sen/Gilbert practice derives the slope interval by inverting the
        # MK test and selecting discrete order statistics of the sorted slope sample.
        # These bounds are therefore not necessarily symmetric around the median slope.
        c_alpha = -spstats.norm.ppf((1.0 - alpha_cl / 100.0) / 2.0) * np.sqrt(
            kendall_variance
        )
        lower_index = max(int(np.round((n_slopes - c_alpha) / 2.0)) - 1, 0)
        upper_index = min(
            int(np.round((n_slopes + c_alpha) / 2.0)), n_slopes - 1
        )

        lcl = float(pairwise_slopes[lower_index])
        ucl = float(pairwise_slopes[upper_index])
        return float(slope), lcl, ucl

    def _compute_pairwise_slopes(
        self,
        obs_dts_valid: np.ndarray,
        obs_valid: np.ndarray,
        strata_labels: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Compute valid pairwise slopes, optionally in parallel."""
        n_obs = obs_valid.size
        if n_obs < 2:
            return []

        chunk_starts = self._pairwise_slope_chunk_starts(n_obs)
        if len(chunk_starts) == 1:
            return self._pairwise_slope_chunk(
                chunk_starts[0],
                n_obs - 1,
                obs_dts_valid,
                obs_valid,
                strata_labels=strata_labels,
            )

        chunks = Parallel(n_jobs=self.n_jobs)(
            delayed(self._pairwise_slope_chunk)(
                start,
                stop,
                obs_dts_valid,
                obs_valid,
                strata_labels=strata_labels,
            )
            for start, stop in zip(chunk_starts[:-1], chunk_starts[1:])
        )
        return [slopes for chunk in chunks for slopes in chunk]

    @staticmethod
    def _pairwise_slope_chunk_starts(n_obs: int) -> list[int]:
        """Return chunk boundaries for pairwise slope generation."""
        n_pairs = n_obs * (n_obs - 1) // 2
        # Keep small problems serial to avoid joblib overhead.
        if n_pairs < 100_000:
            return [0, n_obs - 1]

        chunk_count = min(max(2, n_pairs // 250_000), max(2, n_obs - 1))
        boundaries = np.linspace(0, n_obs - 1, num=chunk_count + 1, dtype=int)
        boundaries = np.unique(boundaries)
        if boundaries[-1] != n_obs - 1:
            boundaries = np.append(boundaries, n_obs - 1)
        return boundaries.tolist()

    def _pairwise_slope_chunk(
        self,
        start: int,
        stop: int,
        obs_dts_valid: np.ndarray,
        obs_valid: np.ndarray,
        strata_labels: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Compute pairwise slopes for a contiguous block of leading indices."""
        slopes: list[np.ndarray] = []
        for index in range(start, stop):
            delta_obs = obs_valid[index + 1 :] - obs_valid[index]
            delta_seconds = self._timedelta64_to_seconds(
                obs_dts_valid[index + 1 :] - obs_dts_valid[index]
            )
            valid_delta = self._pairwise_valid_mask(
                obs_dts_valid,
                index,
                strata_labels=strata_labels,
            )
            if np.any(valid_delta):
                slopes.append(
                    delta_obs[valid_delta] / delta_seconds[valid_delta]
                )
        return slopes

    # ------------------------------------------------------------------
    # Autocorrelation and prewhitening helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
        """Return Pearson correlation, guarding against degenerate inputs."""
        if x.size < 2 or y.size < 2:
            return np.nan
        if np.allclose(x, x[0], equal_nan=False) or np.allclose(
            y, y[0], equal_nan=False
        ):
            return np.nan
        return float(spstats.pearsonr(x, y)[0])

    def _nanautocorr(
        self, obs: np.ndarray, nlags: int, r: int = 0
    ) -> tuple[np.ndarray, float]:
        """Compute lag autocorrelations while ignoring NaNs."""
        obs_corr = obs - np.nanmean(obs)
        autocorr_values: list[float] = []

        for lag in range(1, nlags + 1):
            obs_1 = obs_corr[lag:]
            obs_2 = obs_corr[:-lag]
            mask = ~np.isnan(obs_1) & ~np.isnan(obs_2)
            autocorr_values.append(
                self._safe_pearsonr(obs_1[mask], obs_2[mask])
            )

        mask = ~np.isnan(obs_corr)
        autocorr = np.array(
            [self._safe_pearsonr(obs_corr[mask], obs_corr[mask])]
            + autocorr_values,
            dtype=np.float64,
        )
        bartlett_bound = (
            1.96
            * len(obs) ** (-0.5)
            * np.nansum(autocorr[: r + 1] ** 2) ** 0.5
        )
        return autocorr, float(bartlett_bound)

    @staticmethod
    def _levinson(
        r: np.ndarray, n: int
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Wrap Levinson-Durbin output to match the original convention."""
        out = stattools.levinson_durbin(r, nlags=n, isacov=True)
        return (
            np.array([1.0] + list(-out[1]), dtype=np.float64),
            float(out[0]),
            -out[2][1:],
        )

    def _nanprewhite_arok(
        self, obs: np.ndarray, alpha_ak: float
    ) -> tuple[float, np.ndarray, bool]:
        """Prewhiten a series if lag-1 autocorrelation is significant."""
        data = np.array(obs, dtype=np.float64, copy=True)
        if np.all(np.isnan(data)):
            return (
                np.nan,
                np.zeros(len(data), dtype=np.float64) * np.nan,
                np.nan,
            )

        data[np.isinf(data)] = np.nan
        n_valid = np.count_nonzero(~np.isnan(data))
        if n_valid < 5:
            return np.nan, data, False

        p_ind = 5
        nlag = 10
        if nlag > len(data) / 2:
            nlag = len(data) // 2
        if p_ind >= nlag:
            p_ind = nlag - 1
        if nlag <= 0 or p_ind < 0:
            return np.nan, data, False

        autocorr, _ = self._nanautocorr(data, nlag, p_ind)
        if np.all(np.isnan(autocorr)):
            return np.nan, data, False

        _, _, ak_coefs = self._levinson(autocorr / n_valid, p_ind)
        ak_coefs *= -1
        uconf = spstats.norm.ppf(
            1.0 - (1.0 - alpha_ak / 100.0) / 2.0
        ) / np.sqrt(n_valid)

        ak_lag = float(autocorr[1]) if autocorr.size > 1 else np.nan
        if (
            ak_coefs.size == 0
            or not np.isfinite(ak_coefs[0])
            or np.abs(ak_coefs[0]) < uconf
        ):
            warnings.warn("No statistically significant autocorrelation.")
            return ak_lag, data, False

        y = np.zeros(len(data), dtype=np.float64) * np.nan
        y[1:] = autocorr[1] * data[:-1]
        data_prewhite = data - y
        return ak_lag, data_prewhite, True

    def _prewhite_single_group(
        self,
        obs: np.ndarray,
        obs_dts: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute all prewhitened series for one homogeneous instrument group."""
        data = np.array(obs, dtype=np.float64, copy=True)
        data[np.isinf(data)] = np.nan
        data_pw: dict[str, np.ndarray] = {}

        c_pw, data_ar_removed, ss_ak = self._nanprewhite_arok(
            data,
            alpha_ak=float(self.alpha_ak),
        )
        if not (
            (np.count_nonzero(~np.isnan(data_ar_removed)) > 0)
            and ss_ak
            and np.isfinite(c_pw)
            and self._has_stable_prewhitening_denom(float(c_pw))
        ):
            for key in ("pw", "pw_cor", "tfpw_y", "tfpw_ws", "vctfpw"):
                data_pw[key] = np.array(data, copy=True)
            return data_pw

        data_pw["pw"] = np.array(data_ar_removed, copy=True)
        data_pw["pw_cor"] = data_ar_removed / (1.0 - c_pw)

        # First compute slopes for the corrected PW and original series.
        ties_pw = self._nb_tie(data_pw["pw_cor"])
        _, time_tie_counts_pw = self._s_test(data_pw["pw_cor"], obs_dts)
        var_pw = self._kendall_var(
            data_pw["pw_cor"], ties_pw, time_tie_counts_pw
        )
        b0_pw, _, _ = self._sen_slope(
            obs_dts,
            data_pw["pw_cor"],
            var_pw,
            alpha_cl=90.0,
        )

        ties_or = self._nb_tie(data)
        _, time_tie_counts_or = self._s_test(data, obs_dts)
        var_or = self._kendall_var(data, ties_or, time_tie_counts_or)
        b0_or, _, _ = self._sen_slope(obs_dts, data, var_or, alpha_cl=90.0)

        time_seconds = self._timedelta64_to_seconds(obs_dts - obs_dts[0])
        data_detrend_pw = data - b0_pw * time_seconds
        data_detrend_or = data - b0_or * time_seconds

        c_vctfpw, data_ar_removed_or, _ = self._nanprewhite_arok(
            data_detrend_or,
            alpha_ak=float(self.alpha_ak),
        )
        ak_pw, data_ar_removed_pw, ss_pw = self._nanprewhite_arok(
            data_detrend_pw,
            alpha_ak=float(self.alpha_ak),
        )

        if np.count_nonzero(~np.isnan(data_ar_removed_or)) > 0:
            data_pw["tfpw_y"] = data_ar_removed_or + b0_or * time_seconds
        else:
            data_pw["tfpw_y"] = np.array(data, copy=True)

        # TFPW-WS iterates autocorrelation and slope estimates until they stabilize.
        if np.isfinite(ak_pw) and ss_pw:
            if not self._has_stable_prewhitening_denom(float(ak_pw)):
                warnings.warn(
                    "Skipping unstable TFPW-WS update because lag-1 autocorrelation is too close to 1."
                )
                data_pw["tfpw_ws"] = np.array(data, copy=True)
            else:
                c_previous = float(c_pw)
                data_ar_removed_pw = np.array(data, copy=True)
                data_ar_removed_pw[1:] -= ak_pw * data[:-1]
                data_ar_removed_pw[1:] /= 1.0 - ak_pw

                ties = self._nb_tie(data_ar_removed_pw)
                _, time_tie_counts = self._s_test(data_ar_removed_pw, obs_dts)
                variance = self._kendall_var(
                    data_ar_removed_pw, ties, time_tie_counts
                )
                b1_pw, _, _ = self._sen_slope(
                    obs_dts, data_ar_removed_pw, variance, alpha_cl=90.0
                )

                n_loops = 0
                while (np.abs(ak_pw - c_previous) > 1e-4) and (
                    np.abs(b1_pw - b0_pw) > (1e-4 / 24.0 / 3600.0)
                ):
                    if (
                        np.isfinite(ak_pw)
                        and ss_pw
                    ):
                        n_loops += 1
                        data_detrend_pw = data - b1_pw * time_seconds
                        c_previous = float(ak_pw)
                        b0_pw = float(b1_pw)
                        ak_pw, data_ar_removed2_pw, ss_pw = (
                            self._nanprewhite_arok(
                                data_detrend_pw,
                                alpha_ak=float(self.alpha_ak),
                            )
                        )

                        if (
                            np.isfinite(ak_pw)
                            and (ak_pw > 0.0)
                            and ss_pw
                        ):
                            if not self._has_stable_prewhitening_denom(
                                float(ak_pw)
                            ):
                                warnings.warn(
                                    "Stopping unstable TFPW-WS iteration because lag-1 autocorrelation is too close to 1."
                                )
                                break
                            data_ar_removed2_pw = np.array(data, copy=True)
                            data_ar_removed2_pw[1:] -= ak_pw * data[:-1]
                            data_ar_removed2_pw[1:] /= 1.0 - ak_pw

                            ties = self._nb_tie(data_ar_removed2_pw)
                            _, time_tie_counts = self._s_test(
                                data_ar_removed2_pw, obs_dts
                            )
                            variance = self._kendall_var(
                                data_ar_removed2_pw,
                                ties,
                                time_tie_counts,
                            )
                            b1_pw, _, _ = self._sen_slope(
                                obs_dts,
                                data_ar_removed2_pw,
                                variance,
                                alpha_cl=90.0,
                            )
                            data_ar_removed_pw = np.array(
                                data_ar_removed2_pw, copy=True
                            )

                            if n_loops > 10:
                                break
                    else:
                        break

        if np.count_nonzero(~np.isnan(data_ar_removed_pw)) > 0:
            data_pw["tfpw_ws"] = np.array(data_ar_removed_pw, copy=True)
        else:
            data_pw["tfpw_ws"] = np.array(data, copy=True)

        # VCTFPW rescales the prewhitened residuals and adds a corrected trend back.
        var_data = np.nanvar(data, ddof=1)
        var_data_tfpw = np.nanvar(data_ar_removed_or, ddof=1)
        if np.isfinite(var_data_tfpw) and var_data_tfpw != 0.0:
            data_ar_removed_var = data_ar_removed_or * var_data / var_data_tfpw
        else:
            data_ar_removed_var = np.array(data_ar_removed_or, copy=True)

        if np.isfinite(c_vctfpw) and c_vctfpw >= 0.0 and c_vctfpw < 1.0:
            b_vc = b0_or / np.sqrt((1.0 + c_vctfpw) / (1.0 - c_vctfpw))
        else:
            b_vc = float(b0_or)

        data_pw["vctfpw"] = data_ar_removed_var + b_vc * time_seconds
        return data_pw

    def _prewhite(
        self,
        obs: np.ndarray,
        obs_dts: np.ndarray,
        instrument_labels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute prewhitened series stratified by instrument when available."""
        if not self._use_instrument_prewhitening(obs, instrument_labels):
            return self._prewhite_single_group(obs, obs_dts)

        data_pw = {
            key: np.full(obs.shape, np.nan, dtype=np.float64)
            for key in ("pw", "pw_cor", "tfpw_y", "tfpw_ws", "vctfpw")
        }

        valid_instruments = instrument_labels[~np.isnan(obs)]
        for instrument in np.unique(valid_instruments):
            instrument_mask = instrument_labels == instrument
            prewhitened = self._prewhite_single_group(
                obs[instrument_mask],
                obs_dts[instrument_mask],
            )
            for key, values in prewhitened.items():
                data_pw[key][instrument_mask] = values

        return data_pw

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------
    @staticmethod
    def _prob_3pw(
        p_pw: float, p_tfpw_y: float, alpha_mk: float
    ) -> tuple[float, int]:
        """Combine 3PW probabilities using the original decision rules."""
        p_alpha = 1.0 - alpha_mk / 100.0
        p_value = float(np.nanmax([p_pw, p_tfpw_y]))

        if (p_pw <= p_alpha) and (p_tfpw_y <= p_alpha):
            significance = int(alpha_mk)
        elif (p_pw > p_alpha) and (p_tfpw_y <= p_alpha):
            significance = -1
        elif (p_tfpw_y > p_alpha) and (p_pw <= p_alpha):
            significance = -2
        else:
            significance = 0

        return p_value, significance

    def _compute_mk_stat(
        self,
        obs: np.ndarray,
        alpha_cl: float,
        cache_key: str | None = None,
    ) -> tuple[dict[str, float | int], int, float, float]:
        """Compute MK p-value, significance, slope, and confidence bounds."""
        valid_count = int(np.count_nonzero(~np.isnan(obs)))
        if valid_count < self.MIN_VALID_SAMPLES:
            raise ValueError(
                f"Mann-Kendall trend estimation requires more than 10 valid observations, got {valid_count}."
            )

        if self.stratified_:
            strata_labels = self.strata_labels_
            if strata_labels is None:
                raise RuntimeError(
                    "Stratified MK computation requires fitted strata labels."
                )
            s_value, variance = self._compute_stratified_stats(
                obs,
                self.obs_dts_,
                strata_labels,
            )
        else:
            ties = self._nb_tie(obs)
            s_value, time_tie_counts = self._s_test(obs, self.obs_dts_)
            variance = self._kendall_var(obs, ties, time_tie_counts)
        z_value = self._std_normal_var(s_value, variance)

        # This implementation intentionally uses the asymptotic MK distribution only.
        p_value = 2.0 * (
            1.0 - spstats.norm.cdf(np.abs(z_value), loc=0.0, scale=1.0)
        )
        significance = (
            int(self.alpha_mk)
            if p_value <= 1.0 - float(self.alpha_mk) / 100.0
            else 0
        )

        slope, slope_min, slope_max = self._sen_slope(
            self.obs_dts_,
            obs,
            variance,
            alpha_cl=alpha_cl,
            strata_labels=(self.strata_labels_ if self.stratified_ else None),
            cache_key=cache_key,
        )
        result = {
            "p": float(p_value),
            "ss": significance,
            "slope": float(slope * self.SECONDS_PER_YEAR),
            "ucl": float(slope_max * self.SECONDS_PER_YEAR),
            "lcl": float(slope_min * self.SECONDS_PER_YEAR),
        }
        return result, s_value, variance, z_value

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: Any,
        y: Any | None = None,
        instrument_labels: Any | None = None,
    ):
        """Fit the trend estimator for a single time series."""
        if y is None:
            raise ValueError(
                "fit() requires observation values through the y argument."
            )
        if self.pw_method not in self.VALID_PW_METHODS:
            raise ValueError(f"Unsupported pw_method: {self.pw_method}")

        obs_dts = self._as_datetime64_array(X)
        obs = self._as_float_array(y)
        if obs.size != obs_dts.size:
            raise ValueError(
                "Observation times and values must have the same length."
            )
        instruments = self._as_instrument_array(instrument_labels, obs.size)

        sort_index = np.argsort(obs_dts)
        self.obs_dts_ = obs_dts[sort_index]
        self.obs_ = obs[sort_index]
        self.instrument_labels_ = instruments[sort_index]
        self.season_labels_ = self._season_labels(self.obs_dts_)
        self.strata_labels_ = self._build_strata_labels(
            self.obs_dts_,
            self.instrument_labels_,
        )
        self.stratified_ = self._use_stratification(
            self.obs_,
            self.strata_labels_,
        )

        valid_count = int(np.count_nonzero(~np.isnan(self.obs_)))
        if valid_count < self.MIN_VALID_SAMPLES:
            raise ValueError(
                f"Mann-Kendall trend estimation requires more than 10 valid observations, got {valid_count}."
            )

        self.prewhitened_ = self._prewhite(
            self.obs_,
            self.obs_dts_,
            self.instrument_labels_,
        )
        self._result_cache_: dict[float, dict[str, float | int]] = {}
        self._pairwise_slope_cache_: dict[str, dict[str, Any]] = {}

        default_result = self.get_result(alpha_cl=float(self.alpha_cl))
        self.result_ = default_result
        self.p_value_ = float(default_result["p"])
        self.significance_ = int(default_result["ss"])
        self.slope_ = float(default_result["slope"])
        self.lower_ci_ = float(default_result["lcl"])
        self.upper_ci_ = float(default_result["ucl"])
        self.n_features_in_ = 1
        return self

    def get_result(
        self, alpha_cl: float | None = None
    ) -> dict[str, float | int]:
        """Return the fitted trend result for a chosen confidence level."""
        check_is_fitted(self, "prewhitened_")
        alpha_cl = float(self.alpha_cl if alpha_cl is None else alpha_cl)
        cached = self._result_cache_.get(alpha_cl)
        if cached is not None:
            return dict(cached)

        if self.pw_method == "3pw":
            result_pw, _, _, _ = self._compute_mk_stat(
                self.prewhitened_["pw"],
                alpha_cl=alpha_cl,
                cache_key="pw",
            )
            result_tfpw_y, _, _, _ = self._compute_mk_stat(
                self.prewhitened_["tfpw_y"],
                alpha_cl=alpha_cl,
                cache_key="tfpw_y",
            )
            result_vctfpw, _, _, _ = self._compute_mk_stat(
                self.prewhitened_["vctfpw"],
                alpha_cl=alpha_cl,
                cache_key="vctfpw",
            )
            p_value, significance = self._prob_3pw(
                float(result_pw["p"]),
                float(result_tfpw_y["p"]),
                float(self.alpha_mk),
            )
            result = {
                "p": p_value,
                "ss": significance,
                "slope": float(result_vctfpw["slope"]),
                "ucl": float(result_vctfpw["ucl"]),
                "lcl": float(result_vctfpw["lcl"]),
            }
        else:
            result, _, _, _ = self._compute_mk_stat(
                self.prewhitened_[self.pw_method],
                alpha_cl=alpha_cl,
                cache_key=self.pw_method,
            )

        self._result_cache_[alpha_cl] = dict(result)
        return dict(result)

    def get_trend_slope_quantile_function(
        self,
        n_alpha_points: int = 2001,
        series_key: str | None = None,
    ) -> dict[str, Any]:
        """Return the quantile function of the trend slope point estimate.

        This is the actual sampling distribution implied by the Sen/Gilbert
        confidence-interval construction used in ``_sen_slope``: for a chosen
        confidence level ``alpha_cl``, that method inverts the (asymptotic
        normal approximation to the) Mann-Kendall test statistic to select two
        discrete order statistics of the sorted pairwise-slope sample as the
        lower/upper confidence bounds. This is *not* the same as the empirical
        distribution of the pairwise slopes themselves -- the pairwise-slope
        sample is only used here as a lookup table of order statistics, with
        the mapping from ``alpha_cl`` to order-statistic index governed by the
        normal approximation to the Mann-Kendall S-statistic.

        This method evaluates that mapping at many evenly spaced confidence
        levels to trace out the quantile function (inverse CDF) of the trend
        slope estimate itself::

                for each alpha_cl in linspace(0, 100, n_alpha_points):
                        lcl(alpha_cl), ucl(alpha_cl) = invert MK test (see _sen_slope)
                        lower_p = (100 - alpha_cl) / 200  # in [0, 0.5]
                        upper_p = (100 + alpha_cl) / 200  # in [0.5, 1]

        The resulting ``(probability_levels, quantile_values)`` pairs are the
        quantile function of the trend estimate and are suitable for PDF
        reconstruction (e.g. via inverse-transform sampling + KDE) and for
        Monte Carlo or Bayesian hierarchical downstream regression.

        Each evaluation reuses the pairwise-slope sample cached in
        ``self._pairwise_slope_cache_`` (populated once per series by
        ``_sen_slope``/``_compute_mk_stat``), so tracing out the full quantile
        function only requires O(n_alpha_points) index lookups, not
        recomputation of the O(n^2) pairwise-slope generation.

        Parameters
        ----------
        n_alpha_points : int, default=2001
                Number of evenly spaced confidence levels in (0, 100) at which to
                evaluate the inverted Sen/Gilbert confidence bounds.
        series_key : str | None, default=None
                Which prewhitened series to use. Defaults to the series actually
                used for ``get_result()``'s reported slope/CI: ``"vctfpw"`` when
                ``pw_method == "3pw"``, otherwise ``self.pw_method``.

        Returns
        -------
        dict[str, Any]
                ``probability_levels`` (ndarray, sorted ascending in (0, 1)),
                ``quantile_values`` (ndarray, per year, aligned with
                ``probability_levels``), ``kendall_variance`` (float), ``slope``
                (float, per year, the median), ``series_key`` (str), ``n_slopes``
                (int, size of the underlying pairwise-slope sample).
        """
        check_is_fitted(self, "prewhitened_")
        if series_key is None:
            series_key = (
                "vctfpw" if self.pw_method == "3pw" else self.pw_method
            )

        cached_entry = self._pairwise_slope_cache_.get(series_key)
        if cached_entry is None:
            # Populate the cache for this series. The CI bounds computed here
            # are discarded; only the cached pairwise-slope sample and
            # kendall_variance are used below.
            self._compute_mk_stat(
                self.prewhitened_[series_key],
                alpha_cl=float(self.alpha_cl),
                cache_key=series_key,
            )
            cached_entry = self._pairwise_slope_cache_.get(series_key)

        if cached_entry is None:
            raise ValueError(
                f"Could not compute pairwise slopes for series '{series_key}'; "
                "the series may not have enough valid observations."
            )

        kendall_variance = cached_entry["kendall_variance"]
        strata_labels = self.strata_labels_ if self.stratified_ else None
        obs_series = self.prewhitened_[series_key]

        # Avoid the exact alpha_cl=100 singularity (norm.ppf(0) == -inf).
        epsilon = 1e-6
        alpha_grid = np.linspace(epsilon, 100.0 - epsilon, int(n_alpha_points))

        lower_probabilities = (100.0 - alpha_grid) / 200.0
        upper_probabilities = (100.0 + alpha_grid) / 200.0

        lcl_values = np.empty(alpha_grid.size, dtype=np.float64)
        ucl_values = np.empty(alpha_grid.size, dtype=np.float64)
        for index, alpha_cl in enumerate(alpha_grid):
            _, lcl, ucl = self._sen_slope(
                self.obs_dts_,
                obs_series,
                kendall_variance,
                alpha_cl=float(alpha_cl),
                strata_labels=strata_labels,
                cache_key=series_key,
            )
            lcl_values[index] = lcl
            ucl_values[index] = ucl

        probability_levels = np.concatenate(
            [lower_probabilities, upper_probabilities, [0.5]]
        )
        quantile_values = (
            np.concatenate([lcl_values, ucl_values, [cached_entry["slope"]]])
            * self.SECONDS_PER_YEAR
        )

        sort_index = np.argsort(probability_levels)
        probability_levels = probability_levels[sort_index]
        quantile_values = quantile_values[sort_index]

        return {
            "probability_levels": probability_levels,
            "quantile_values": quantile_values,
            "n_slopes": int(cached_entry["n_slopes"]),
            "kendall_variance": float(kendall_variance),
            "slope": float(cached_entry["slope"]) * self.SECONDS_PER_YEAR,
            "series_key": series_key,
        }
