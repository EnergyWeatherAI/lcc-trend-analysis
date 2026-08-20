import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils.validation import check_is_fitted
import statsmodels.api as sm
from joblib import Parallel, delayed


class CloudFeedbackGLM(BaseEstimator, RegressorMixin):
    """
    Quasi-binomial GLM with seasonal smoothing and stationary block
    bootstrap confidence intervals (parallelized).
    Predictors: T_trend (long-term) and T_anom (short-term) in K.
    """

    _parameter_constraints = {
        "seasonal_df": [int],
        "block_size": [int],
        "n_bootstrap": [int],
        "alpha": [float],
        "n_jobs": [int],
        "include_interaction": [bool],
        "random_state": [int, type(None)],
    }

    def __init__(
        self,
        seasonal_df=12,
        block_size=60,
        n_bootstrap=1000,
        alpha=0.05,
        n_jobs=-1,
        include_interaction=False,
        random_state=None,
    ):
        self.seasonal_df = seasonal_df
        self.block_size = block_size
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.include_interaction = include_interaction
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _make_seasonal_basis(self, doy):
        """Fourier basis for seasonal variation (period = 365.25 days).

        Produces ``seasonal_df`` columns: pairs of sin/cos for each
        harmonic.  No data-dependent knots — the basis is identical
        for any DOY input, so there is no state to store or pickle.
        """
        doy = np.asarray(doy, dtype=float)
        omega = 2.0 * np.pi / 365.25
        n_harmonics = self.seasonal_df // 2
        cols = []
        for k in range(1, n_harmonics + 1):
            cols.append(np.sin(k * omega * doy))
            cols.append(np.cos(k * omega * doy))
        if self.seasonal_df % 2 == 1:
            cols.append(np.sin((n_harmonics + 1) * omega * doy))
        return np.column_stack(cols)

    def _make_design_matrix(self, T_trend, T_anom, doy):
        """Construct full GLM design matrix, optionally including temp×season interaction.

        Column layout:
            0          : constant
            1          : T_trend
            2          : T_anom
            3 .. 3+S-1 : seasonal basis  (S columns)
          If ``include_interaction``:
            3+S .. 3+2S-1  : T_trend × seasonal
            3+2S .. 3+3S-1 : T_anom  × seasonal
        """
        T_trend = np.asarray(T_trend).reshape(-1, 1)
        T_anom = np.asarray(T_anom).reshape(-1, 1)
        seasonal = self._make_seasonal_basis(doy)

        # Store column count for MME computation
        self._n_seasonal_cols_ = seasonal.shape[1]

        X_list = [T_trend, T_anom, seasonal]

        if self.include_interaction:
            X_list.append(T_trend * seasonal)
            X_list.append(T_anom * seasonal)

        X = np.hstack(X_list)
        X = np.column_stack([np.ones(X.shape[0]), X])
        return X

    def _fit_glm(self, X, y, n):
        """Fit quasi-binomial GLM"""
        np_errstate = np.geterr()
        np.seterr(all="ignore")  # Ignore warnings from statsmodels
        y_clipped = np.clip(y, 1e-2, 1 - 1e-2)
        model = sm.GLM(y_clipped, X, family=sm.families.Binomial(), var_weights=n)
        result = model.fit(scale="X2")  # quasi-binomial dispersion
        np.seterr(**np_errstate)  # Restore original error handling
        return result

    # ------------------------------------------------------------------
    # Stationary bootstrap
    # ------------------------------------------------------------------
    def _stationary_bootstrap_indices(self, n_samples, rng):
        """Generate indices for stationary bootstrap (Politis & Romano, 1994).
        
        Builds a resampled index array by concatenating blocks of consecutive
        observations.  Block lengths are i.i.d. Geometric(1/block_size), so
        the expected block length equals ``self.block_size``.  Indices wrap
        around circularly so every starting position is equally likely.
        """
        indices = []
        p = 1 / self.block_size   # geometric probability
        while len(indices) < n_samples:
            start = rng.integers(0, n_samples)
            k = rng.geometric(p)
            block = np.arange(start, start + k) % n_samples  # circular wrap
            indices.extend(block)
        return np.array(indices[:n_samples])
    
    def _compute_mme(self, result, X_design, p_pred):
        """Compute mean marginal effects for T_trend and T_anom.

        When ``include_interaction`` is True the marginal effect of, e.g.,
        T_trend is ``(β_trend + Σ_j β_{trend×season,j} · S_j) · p · (1-p)``
        rather than just ``β_trend · p · (1-p)``.
        """
        params = result.params
        beta_trend = params[1]
        beta_anom = params[2]

        if self.include_interaction:
            S = self._n_seasonal_cols_
            # Seasonal basis columns in X_design (after const, T_trend, T_anom)
            seasonal_cols = X_design[:, 3 : 3 + S]
            # Interaction blocks start after [const, T_trend, T_anom, seasonal]
            interaction_start = 3 + S
            beta_trend_x_season = params[interaction_start : interaction_start + S]
            beta_anom_x_season = params[interaction_start + S : interaction_start + 2 * S]

            me_trend = (beta_trend + seasonal_cols @ beta_trend_x_season) * p_pred * (1 - p_pred)
            me_anom = (beta_anom + seasonal_cols @ beta_anom_x_season) * p_pred * (1 - p_pred)
        else:
            me_trend = beta_trend * p_pred * (1 - p_pred)
            me_anom = beta_anom * p_pred * (1 - p_pred)

        return np.mean(me_trend), np.mean(me_anom)

    def _bootstrap_single(self, T, time, doy, y, n, rng):
        """Single bootstrap iteration using case resampling.

        Block-resamples all covariates and response jointly, uses the
        original temperature trend decomposition, and refits the GLM.
        """
        idx = self._stationary_bootstrap_indices(len(y), rng)

        Tb = T[idx]
        tb = time[idx]
        doyb = doy[idx]
        yb = y[idx]
        nb = n[idx]

        # Recompute trend decomposition for this resample
        coef = np.polyfit(tb, Tb, 1)
        T_trend_b = coef[0] * tb + coef[1]
        T_anom_b = Tb - T_trend_b

        try:
            Xb = self._make_design_matrix(T_trend_b, T_anom_b, doyb)
            res = self._fit_glm(Xb, yb, nb)
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

        np_errstate = np.geterr()
        np.seterr(all="ignore")
        p_pred = res.predict(Xb)
        np.seterr(**np_errstate)

        mme_trend, mme_anom = self._compute_mme(res, Xb, p_pred)

        return res.params[1], res.params[2], mme_trend, mme_anom

    def _run_bootstrap(self, T, time, doy, y, n):
        """Run stationary block bootstrap with case resampling.

        Each iteration block-resamples all variables jointly, recomputes
        the temperature trend/anomaly decomposition, and refits the GLM.
        """
        rng_base = np.random.default_rng(self.random_state)
        seeds = rng_base.integers(0, 2**32 - 1, size=self.n_bootstrap)

        # Temporarily remove unpicklable attributes for joblib serialisation
        _stashed: dict = {}
        for attr in ("result_",):
            if attr in self.__dict__:
                _stashed[attr] = self.__dict__.pop(attr)

        try:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._bootstrap_single)(
                    T, time, doy, y, n,
                    np.random.default_rng(seed)
                )
                for seed in seeds
            )
        finally:
            self.__dict__.update(_stashed)

        results = np.array(results)

        # Filter out failed resamples
        valid = ~np.isnan(results[:, 0])
        n_valid = valid.sum()
        results = results[valid]

        if n_valid == 0:
            import warnings
            warnings.warn(
                f"All {self.n_bootstrap} bootstrap resamples failed. "
                f"Returning NaN arrays. This typically indicates extreme "
                f"cloud fractions or insufficient data.",
                RuntimeWarning,
            )
            empty = np.array([np.nan])
            return empty, empty, empty, empty

        if n_valid < self.n_bootstrap * 0.5:
            import warnings
            warnings.warn(
                f"Only {n_valid}/{self.n_bootstrap} bootstrap resamples "
                f"converged. CIs may be unreliable.",
                RuntimeWarning,
            )

        slopes_trend, slopes_anom = results[:, 0], results[:, 1]
        mme_trend_vals, mme_anom_vals = results[:, 2], results[:, 3]

        return slopes_trend, slopes_anom, mme_trend_vals, mme_anom_vals

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        df = X.copy()

        # Ensure temporal ordering for block bootstrap
        sort_idx = df["time"].values.argsort()
        df = df.iloc[sort_idx].reset_index(drop=True)
        y = np.asarray(y)[sort_idx]

        T = df["temperature"].values
        doy = df["doy"].values
        n = df["n_samples"].values
        time = df["time"].values

        # Decompose temperature into trend + anomaly
        trend_coef = np.polyfit(time, T, 1)
        T_trend = trend_coef[0] * time + trend_coef[1]
        T_anom = T - T_trend

        # Store decomposition for later use
        self.trend_coef_ = trend_coef

        X_design = self._make_design_matrix(T_trend, T_anom, doy)

        if X_design.shape[0] < X_design.shape[1]:
            raise ValueError(
                f"Insufficient observations ({X_design.shape[0]}) for "
                f"{X_design.shape[1]} design matrix columns."
            )

        self.result_ = self._fit_glm(X_design, y, n)

        # Fitted values and MME
        np_errstate = np.geterr()
        np.seterr(all="ignore")
        p_pred = self.result_.predict(X_design)
        np.seterr(**np_errstate)

        self.mme_trend_, self.mme_anom_ = self._compute_mme(
            self.result_, X_design, p_pred
        )

        # Case resampling bootstrap
        (
            slopes_trend,
            slopes_anom,
            mme_trend_vals,
            mme_anom_vals,
        ) = self._run_bootstrap(T, time, doy, y, n)

        self.bootstrap_slopes_ = np.column_stack([slopes_trend, slopes_anom])
        self.bootstrap_mme_ = np.column_stack([mme_trend_vals, mme_anom_vals])

        self.X_design_ = X_design
        self.n_features_in_ = X_design.shape[1]

        return self

    def predict(self, X):
        check_is_fitted(self, "result_")
        df = X.copy()
        T = df["temperature"].values
        doy = df["doy"].values
        time = df["time"].values

        T_trend = self.trend_coef_[0] * time + self.trend_coef_[1]
        T_anom = T - T_trend

        X_design = self._make_design_matrix(T_trend, T_anom, doy)
        
        np_errstate = np.geterr()
        np.seterr(all="ignore") # Ignore warnings from statsmodels
        res = self.result_.predict(X_design)
        np.seterr(**np_errstate)  # Restore original error handling
        return res

    def get_bootstrap_ci(self, alpha=None):
        """Compute confidence intervals from the stored bootstrap distribution.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the CI (e.g. 0.05 for 95% CI).
            Defaults to ``self.alpha``.

        Returns
        -------
        dict
            Confidence intervals for slopes and MME (both trend and anomaly).
        """
        check_is_fitted(self, "result_")
        if alpha is None:
            alpha = self.alpha

        long_mme = self.bootstrap_mme_[:, 0]
        short_mme = self.bootstrap_mme_[:, 1]
        long_slope = self.bootstrap_slopes_[:, 0]
        short_slope = self.bootstrap_slopes_[:, 1]

        # Filter out NaN values for each array
        long_slope_valid = long_slope[~np.isnan(long_slope)]
        short_slope_valid = short_slope[~np.isnan(short_slope)]
        long_mme_valid = long_mme[~np.isnan(long_mme)]
        short_mme_valid = short_mme[~np.isnan(short_mme)]

        lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

        def _safe_percentile(arr, pcts):
            if len(arr) == 0:
                return (np.nan, np.nan)
            return tuple(np.percentile(arr, pcts))

        return {
            "longterm_slope_ci": _safe_percentile(long_slope_valid, [lo, hi]),
            "shortterm_slope_ci": _safe_percentile(short_slope_valid, [lo, hi]),
            "longterm_mme_ci": _safe_percentile(long_mme_valid, [lo, hi]),
            "shortterm_mme_ci": _safe_percentile(short_mme_valid, [lo, hi]),
        }

    def get_mme_statistics(self, alpha=None):
        """Return MME estimates, confidence intervals, and p-values from the bootstrap distribution.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the CI (e.g. 0.05 for 95% CI).
            Defaults to ``self.alpha``.

        Returns
        -------
        dict
        """
        check_is_fitted(self, "result_")
        if alpha is None:
            alpha = self.alpha

        long_mme_vals = self.bootstrap_mme_[:, 0]
        short_mme_vals = self.bootstrap_mme_[:, 1]

        # Filter NaNs for p-value computation
        long_mme_valid = long_mme_vals[~np.isnan(long_mme_vals)]
        short_mme_valid = short_mme_vals[~np.isnan(short_mme_vals)]

        ci = self.get_bootstrap_ci(alpha=alpha)

        def _safe_pvalue(vals):
            if len(vals) == 0:
                return np.nan
            return min(1.0, 2 * min(float(np.mean(vals <= 0)), float(np.mean(vals >= 0))))

        return {
            "longterm_mme": np.nanmean(long_mme_valid),
            "longterm_mme_ci_lower": ci["longterm_mme_ci"][0],
            "longterm_mme_ci_upper": ci["longterm_mme_ci"][1],
            "longterm_mme_p_value": _safe_pvalue(long_mme_valid),

            "shortterm_mme": np.nanmean(short_mme_valid),
            "shortterm_mme_ci_lower": ci["shortterm_mme_ci"][0],
            "shortterm_mme_ci_upper": ci["shortterm_mme_ci"][1],
            "shortterm_mme_p_value": _safe_pvalue(short_mme_valid),
        }


class MetaCloudFeedbackGLM(BaseEstimator, RegressorMixin):
    """Per-instrument CloudFeedbackGLM with combined bootstrap distributions.

    Fits a separate :class:`CloudFeedbackGLM` for each instrument present in
    the data, allocating bootstrap iterations proportional to each
    instrument's sample size.  The per-instrument bootstrap distributions are
    concatenated so that instruments with more observations are naturally
    weighted more heavily in the combined percentile CIs.

    Handles single-instrument sites transparently (delegates to one GLM).
    """

    _parameter_constraints = {
        "seasonal_df": [int],
        "block_size": [int],
        "n_bootstrap": [int],
        "alpha": [float],
        "n_jobs": [int],
        "include_interaction": [bool],
        "random_state": [int, type(None)],
        "min_bootstrap": [int],
    }

    def __init__(
        self,
        seasonal_df=2,
        block_size=60,
        n_bootstrap=10000,
        alpha=0.05,
        n_jobs=-1,
        include_interaction=False,
        random_state=None,
        min_bootstrap=100,
    ):
        self.seasonal_df = seasonal_df
        self.block_size = block_size
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.include_interaction = include_interaction
        self.random_state = random_state
        self.min_bootstrap = min_bootstrap

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        df = X.copy()
        y = np.asarray(y)

        instruments = df["instrument"].unique()
        total_n = len(df)

        # Generate independent random seeds for each instrument
        rng = np.random.default_rng(self.random_state)
        inst_seeds = {
            inst: int(rng.integers(0, 2**32 - 1)) for inst in instruments
        }

        self.estimators_ = {}
        bootstrap_mme_parts = []
        bootstrap_slopes_parts = []
        mme_trends = []
        mme_anoms = []
        weights = []

        for inst in instruments:
            mask = df["instrument"] == inst
            sub_df = df.loc[mask].drop(columns=["instrument"]).copy()
            sub_y = y[mask.values]

            n_inst = int(mask.sum())
            n_boot = max(
                self.min_bootstrap,
                round(self.n_bootstrap * n_inst / total_n),
            )

            glm = CloudFeedbackGLM(
                seasonal_df=self.seasonal_df,
                block_size=self.block_size,
                n_bootstrap=n_boot,
                alpha=self.alpha,
                n_jobs=self.n_jobs,
                include_interaction=self.include_interaction,
                random_state=inst_seeds[inst],
            )
            glm.fit(sub_df, sub_y)
            self.estimators_[inst] = glm

            bootstrap_mme_parts.append(glm.bootstrap_mme_)
            bootstrap_slopes_parts.append(glm.bootstrap_slopes_)
            mme_trends.append(glm.mme_trend_)
            mme_anoms.append(glm.mme_anom_)
            weights.append(n_inst)

        # Combined bootstrap distributions (weighted by representation)
        self.bootstrap_mme_ = np.vstack(bootstrap_mme_parts)
        self.bootstrap_slopes_ = np.vstack(bootstrap_slopes_parts)

        # Sample-size-weighted point estimates
        w = np.array(weights, dtype=float)
        w /= w.sum()
        self.mme_trend_ = float(np.dot(w, mme_trends))
        self.mme_anom_ = float(np.dot(w, mme_anoms))

        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        df = X.copy()
        predictions = np.full(len(df), np.nan)

        for inst, glm in self.estimators_.items():
            mask = df["instrument"] == inst
            if not mask.any():
                continue
            sub_df = df.loc[mask].drop(columns=["instrument"]).copy()
            predictions[mask.values] = glm.predict(sub_df)

        return predictions

    def get_bootstrap_ci(self, alpha=None):
        """Compute confidence intervals from the combined bootstrap distribution.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the CI (e.g. 0.05 for 95% CI).
            Defaults to ``self.alpha``.

        Returns
        -------
        dict
            Confidence intervals for slopes and MME (both trend and anomaly).
        """
        check_is_fitted(self, "estimators_")
        if alpha is None:
            alpha = self.alpha

        long_mme = self.bootstrap_mme_[:, 0]
        short_mme = self.bootstrap_mme_[:, 1]
        long_slope = self.bootstrap_slopes_[:, 0]
        short_slope = self.bootstrap_slopes_[:, 1]

        long_slope_valid = long_slope[~np.isnan(long_slope)]
        short_slope_valid = short_slope[~np.isnan(short_slope)]
        long_mme_valid = long_mme[~np.isnan(long_mme)]
        short_mme_valid = short_mme[~np.isnan(short_mme)]

        lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

        def _safe_percentile(arr, pcts):
            if len(arr) == 0:
                return (np.nan, np.nan)
            return tuple(np.percentile(arr, pcts))

        return {
            "longterm_slope_ci": _safe_percentile(long_slope_valid, [lo, hi]),
            "shortterm_slope_ci": _safe_percentile(short_slope_valid, [lo, hi]),
            "longterm_mme_ci": _safe_percentile(long_mme_valid, [lo, hi]),
            "shortterm_mme_ci": _safe_percentile(short_mme_valid, [lo, hi]),
        }

    def get_mme_statistics(self, alpha=None):
        """Return MME estimates, CIs and p-values from the combined bootstrap.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the CI (e.g. 0.05 for 95% CI).
            Defaults to ``self.alpha``.

        Returns
        -------
        dict
        """
        check_is_fitted(self, "estimators_")
        if alpha is None:
            alpha = self.alpha

        long_mme_vals = self.bootstrap_mme_[:, 0]
        short_mme_vals = self.bootstrap_mme_[:, 1]

        long_mme_valid = long_mme_vals[~np.isnan(long_mme_vals)]
        short_mme_valid = short_mme_vals[~np.isnan(short_mme_vals)]

        ci = self.get_bootstrap_ci(alpha=alpha)

        def _safe_pvalue(vals):
            if len(vals) == 0:
                return np.nan
            return min(1.0, 2 * min(float(np.mean(vals <= 0)), float(np.mean(vals >= 0))))

        return {
            "longterm_mme": np.nanmean(long_mme_valid),
            "longterm_mme_ci_lower": ci["longterm_mme_ci"][0],
            "longterm_mme_ci_upper": ci["longterm_mme_ci"][1],
            "longterm_mme_p_value": _safe_pvalue(long_mme_valid),

            "shortterm_mme": np.nanmean(short_mme_valid),
            "shortterm_mme_ci_lower": ci["shortterm_mme_ci"][0],
            "shortterm_mme_ci_upper": ci["shortterm_mme_ci"][1],
            "shortterm_mme_p_value": _safe_pvalue(short_mme_valid),
        }