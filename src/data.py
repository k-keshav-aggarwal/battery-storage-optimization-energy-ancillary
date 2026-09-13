"""Load cached CAISO prices, aggregate to an hourly series, split by time.

Two fixes relative to the notebook lineage:

1. Timestamps are converted from UTC to Pacific. The cached frames store
   tz-naive UTC (``2023-01-01 08:00`` is midnight PST), so every hour-of-day
   and rolling-24h feature computed on the raw column was shifted by 8 hours
   relative to the market day it claims to describe.

2. Aggregation to a single hourly series happens *before* anything else
   touches the data, so no downstream step can accidentally index into the
   long (3-rows-per-hour) layout by position.
"""
from __future__ import annotations

import os

import pandas as pd

from .config import CACHE_DIR, TRAIN_END, VAL_END

PRICE_COLS = ["SP15", "NonSpin", "RegDown", "RegUp", "Spin"]
_MARKET_TZ = "America/Los_Angeles"


def _to_pacific(s: pd.Series) -> pd.Series:
    """Reinterpret tz-naive UTC timestamps in the CAISO market timezone."""
    dt = pd.to_datetime(s)
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    return dt.dt.tz_convert(_MARKET_TZ).dt.tz_localize(None)


def load_long(cache_dir: str = CACHE_DIR, filename: str = "merged_df_clean.csv.gz") -> pd.DataFrame:
    """Read the cached node-level frame (3 rows per hour, one per node)."""
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Expected the cached CAISO frame; run pull_prices.py "
            f"or place the .csv.gz in {cache_dir}/."
        )
    df = pd.read_csv(path, compression="gzip")
    df["datetime"] = _to_pacific(df["datetime"])
    return df.sort_values(["datetime", "node"]).reset_index(drop=True)


def to_hourly(df_long: pd.DataFrame) -> pd.DataFrame:
    """Collapse the node dimension to one mean price per hour.

    Returned on a gap-free hourly index so that rolling windows measure real
    elapsed time rather than row counts.
    """
    agg = {c: "mean" for c in PRICE_COLS if c in df_long.columns}
    hourly = df_long.groupby("datetime").agg(agg).reset_index()
    hourly = hourly.sort_values("datetime").reset_index(drop=True)

    full_idx = pd.date_range(hourly["datetime"].min(), hourly["datetime"].max(), freq="h")
    hourly = (hourly.set_index("datetime")
                    .reindex(full_idx)
                    .interpolate(limit_direction="both")
                    .rename_axis("datetime")
                    .reset_index())
    return hourly


def per_node_hourly(df_long: pd.DataFrame) -> pd.DataFrame:
    """Wide frame of one price column per node, for the transfer experiment."""
    wide = df_long.pivot_table(index="datetime", columns="node", values="SP15", aggfunc="mean")
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(full_idx).interpolate(limit_direction="both").rename_axis("datetime").reset_index()


def split(df: pd.DataFrame, dt_col: str = "datetime"):
    """Chronological train / val / test split on fixed calendar boundaries."""
    train = df[df[dt_col] < TRAIN_END].reset_index(drop=True)
    val = df[(df[dt_col] >= TRAIN_END) & (df[dt_col] < VAL_END)].reset_index(drop=True)
    test = df[df[dt_col] >= VAL_END].reset_index(drop=True)
    return train, val, test


def split_masks(df: pd.DataFrame, dt_col: str = "datetime"):
    """Boolean masks for the same split, for use on aligned arrays."""
    d = df[dt_col]
    return (d < TRAIN_END).values, ((d >= TRAIN_END) & (d < VAL_END)).values, (d >= VAL_END).values


def load_hourly(cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Convenience: cached file -> clean hourly series."""
    return to_hourly(load_long(cache_dir))
