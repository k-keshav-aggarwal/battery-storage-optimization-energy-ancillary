"""Strictly causal features for genuine-vs-synthetic price classification.

Every feature at hour *t* is a function of prices at hours <= *t* only.

The notebook lineage did not hold to this. Its three highest-importance
features -- ``f_zscore`` (0.217), ``f_persistence`` (0.169) and
``f_non_reversion`` (0.161), together 55% of the model's importance mass --
were all built from ``rolling(center=True)`` or ``shift(-window)``, so they
read between 3 and 12 hours into the future and then drove a dispatch
decision at *t*. It also normalised several features by statistics
(``price_vs_fut.min()/.max()``, ``roc.mean()``) computed over the entire
series including the evaluation window.

Both classes of leak are removed here. Features are additionally built to be
scale-free (ratios, robust z-scores, normalised counts) so that no global
fitted constant is needed at all -- there is nothing left to leak through.

The cost of causality is real and is the point: ``f_non_reversion`` asked
"did this spike revert?", which cannot be known at decision time. Whatever
performance is lost by dropping it is the honest price of operating in real
time, and is reported rather than recovered by peeking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_WINDOW = 24        # one day
LONG_WINDOW = 168        # one week
HOD_WINDOW_DAYS = 14     # same-hour-of-day reference

FEATURE_COLS = [
    "f_robust_z",
    "f_ratio_short",
    "f_ratio_long",
    "f_hod_dev",
    "f_ramp",
    "f_accel",
    "f_persistence",
    "f_run_length",
    "f_vol_ratio",
    "f_level_pct",
    "f_hour_sin",
    "f_hour_cos",
    "f_dow_sin",
    "f_dow_cos",
    "f_vae_score",
    "f_iso_score",
    "f_lof_score",
]

_EPS = 1e-6


def _trailing_median(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 8)).median()


def _trailing_mad(s: pd.Series, window: int) -> pd.Series:
    med = _trailing_median(s, window)
    return (s - med).abs().rolling(window, min_periods=max(2, window // 8)).median()


def build_causal_features(datetimes, price_observed: np.ndarray,
                          detector_scores: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Assemble the causal feature frame.

    Parameters
    ----------
    datetimes : array-like of hourly timestamps, ascending and gap-free.
    price_observed : the schedule the operator sees, attacked or not.
    detector_scores : optional ``{"vae": ..., "iso": ..., "lof": ...}`` arrays,
        already normalised using train-split statistics only.
    """
    dt = pd.to_datetime(pd.Series(datetimes)).reset_index(drop=True)
    p = pd.Series(np.asarray(price_observed, dtype=float)).reset_index(drop=True)
    out = pd.DataFrame({"datetime": dt})

    # --- level relative to recent history (trailing windows include t) ------
    med_s = _trailing_median(p, SHORT_WINDOW)
    med_l = _trailing_median(p, LONG_WINDOW)
    mad_l = _trailing_mad(p, LONG_WINDOW).replace(0, np.nan)

    out["f_robust_z"] = ((p - med_l) / (1.4826 * mad_l + _EPS)).clip(-10, 10)
    out["f_ratio_short"] = (p / (med_s.abs() + 1.0)).clip(-5, 20)
    out["f_ratio_long"] = (p / (med_l.abs() + 1.0)).clip(-5, 20)

    # --- same-hour-of-day reference, strictly from previous days ------------
    # shift(1) first so hour t never contributes to its own reference.
    hod_ref = (p.shift(1)
                .groupby(dt.dt.hour)
                .transform(lambda x: x.rolling(HOD_WINDOW_DAYS, min_periods=3).median()))
    out["f_hod_dev"] = (p / (hod_ref.abs() + 1.0)).clip(-5, 20)

    # --- dynamics -----------------------------------------------------------
    scale = (mad_l * 1.4826 + 1.0)
    out["f_ramp"] = (p.diff() / scale).clip(-10, 10)
    out["f_accel"] = (p.diff().diff() / scale).clip(-10, 10)

    elevated = (p > med_l + 1.4826 * mad_l).astype(float)
    out["f_persistence"] = elevated.rolling(SHORT_WINDOW, min_periods=1).mean()

    # Consecutive elevated hours ending at t, capped and normalised.
    grp = (elevated != elevated.shift()).cumsum()
    out["f_run_length"] = (elevated.groupby(grp).cumsum() / SHORT_WINDOW).clip(0, 1)

    std_s = p.rolling(SHORT_WINDOW, min_periods=4).std()
    std_l = p.rolling(LONG_WINDOW, min_periods=8).std()
    out["f_vol_ratio"] = (std_s / (std_l + _EPS)).clip(0, 10)

    # Percentile of the current price within the trailing week.
    out["f_level_pct"] = p.rolling(LONG_WINDOW, min_periods=8).rank(pct=True)

    # --- calendar -----------------------------------------------------------
    hour, dow = dt.dt.hour, dt.dt.dayofweek
    out["f_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["f_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["f_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["f_dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- unsupervised detector opinions -------------------------------------
    scores = detector_scores or {}
    for name in ("vae", "iso", "lof"):
        arr = scores.get(name)
        out[f"f_{name}_score"] = np.zeros(len(out)) if arr is None else np.asarray(arr, dtype=float)

    return out[["datetime"] + FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def expected_price_causal(datetimes, price_observed: np.ndarray) -> np.ndarray:
    """A causal 'what this hour normally costs' estimate.

    Blends the trailing same-hour-of-day median with the trailing weekly
    median. Used as the fallback price when the defence decides not to
    believe a printed value, and as a robust de-spiked reference.
    """
    dt = pd.to_datetime(pd.Series(datetimes)).reset_index(drop=True)
    p = pd.Series(np.asarray(price_observed, dtype=float)).reset_index(drop=True)

    hod = (p.shift(1)
            .groupby(dt.dt.hour)
            .transform(lambda x: x.rolling(HOD_WINDOW_DAYS, min_periods=2).median()))
    week = _trailing_median(p, LONG_WINDOW)

    est = (0.7 * hod + 0.3 * week)
    return est.ffill().bfill().fillna(float(np.median(p))).values
