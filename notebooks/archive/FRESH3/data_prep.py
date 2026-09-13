"""Real-data loading, aggregation, and chronological split (no synthetic injection)."""
import pandas as pd, numpy as np, os

# Where your CAISO cache pickles live. Override with:
#   export CAISO_DATA_DIR=/path/to/your/pickles
# before running, or just drop the pickles in ./data (relative to wherever you
# run this from -- a script's own directory if run as `python3 data_prep.py`,
# or the notebook's working directory if this code runs as a notebook cell).
UPLOAD_DIR = os.environ.get("CAISO_DATA_DIR", "./data")
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

def load_raw():
    lmp = pd.read_pickle(os.path.join(UPLOAD_DIR, "cache_lmp_20230101_20251231_0c3247.pkl"))
    as_p = pd.read_pickle(os.path.join(UPLOAD_DIR, "cache_as_20230101_20251231.pkl"))
    return lmp, as_p

def build_aggregated():
    lmp, as_p = load_raw()
    df = pd.merge(lmp, as_p, on="datetime", how="inner")
    agg = (df.groupby("datetime")
             .agg({"SP15": "mean", "NonSpin": "mean", "RegDown": "mean", "RegUp": "mean", "Spin": "mean"})
             .reset_index()
             .sort_values("datetime")
             .reset_index(drop=True))
    return agg

def build_per_node():
    lmp, as_p = load_raw()
    df = pd.merge(lmp, as_p, on="datetime", how="inner")
    return df.sort_values(["node", "datetime"]).reset_index(drop=True)

# Chronological split: ~24mo train / 6mo val / 6mo test, aligned to calendar
# boundaries so the cutoffs are auditable, not just "80/10/10 by row count".
TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END   = pd.Timestamp("2025-07-01")

def split(df, dt_col="datetime"):
    train = df[df[dt_col] < TRAIN_END].reset_index(drop=True)
    val   = df[(df[dt_col] >= TRAIN_END) & (df[dt_col] < VAL_END)].reset_index(drop=True)
    test  = df[df[dt_col] >= VAL_END].reset_index(drop=True)
    return train, val, test

# Second, rolling-origin window: cutoffs shifted back 6 months. Used to check
# whether the detector comparison in the paper is specific to the Jul-Dec
# 2025 test period or holds up on a different held-out window too.
TRAIN_END_W2 = pd.Timestamp("2024-07-01")
VAL_END_W2   = pd.Timestamp("2025-01-01")

def split_w2(df, dt_col="datetime"):
    train = df[df[dt_col] < TRAIN_END_W2].reset_index(drop=True)
    val   = df[(df[dt_col] >= TRAIN_END_W2) & (df[dt_col] < VAL_END_W2)].reset_index(drop=True)
    test  = df[(df[dt_col] >= VAL_END_W2) & (df[dt_col] < VAL_END)].reset_index(drop=True)
    return train, val, test

if __name__ == "__main__":
    agg = build_aggregated()
    agg.to_pickle(os.path.join(OUT_DIR, "agg_hourly.pkl"))
    train, val, test = split(agg)
    print(f"Total hours: {len(agg)}  range: {agg.datetime.min()} -> {agg.datetime.max()}")
    print(f"Train: {len(train)} hours ({train.datetime.min()} -> {train.datetime.max()})  ~{len(train)/730:.1f} mo")
    print(f"Val:   {len(val)} hours ({val.datetime.min()} -> {val.datetime.max()})  ~{len(val)/730:.1f} mo")
    print(f"Test:  {len(test)} hours ({test.datetime.min()} -> {test.datetime.max()})  ~{len(test)/730:.1f} mo")
    print()
    print("Known real events visible in test/val windows:")
    print(test.nlargest(3, "SP15")[["datetime","SP15"]])
    print(val.nlargest(3, "SP15")[["datetime","SP15"]])
