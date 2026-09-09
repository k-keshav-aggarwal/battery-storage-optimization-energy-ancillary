import pandas as pd
import os

UPLOAD_DIR = "./data_cache"
OUT_DIR = "./artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

def load_raw():
    lmp = pd.read_csv(os.path.join(UPLOAD_DIR, "merged_df_clean.csv.gz"))
    as_p = lmp.copy()
    if 'datetime' not in lmp.columns and 'Datetime' in lmp.columns:
        lmp = lmp.rename(columns={'Datetime': 'datetime'})
    lmp['datetime'] = pd.to_datetime(lmp['datetime'])
    as_p['datetime'] = lmp['datetime']
    return lmp, as_p

def build_aggregated():
    lmp, as_p = load_raw()
    # Since both are the same, just use lmp directly
    # Group by datetime and average across all nodes
    agg = (lmp.groupby("datetime")
             .agg({"SP15": "mean", "NonSpin": "mean", "RegDown": "mean", "RegUp": "mean", "Spin": "mean"})
             .reset_index()
             .sort_values("datetime")
             .reset_index(drop=True))
    return agg

TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END   = pd.Timestamp("2025-07-01")

def split(df, dt_col="datetime"):
    train = df[df[dt_col] < TRAIN_END].reset_index(drop=True)
    val   = df[(df[dt_col] >= TRAIN_END) & (df[dt_col] < VAL_END)].reset_index(drop=True)
    test  = df[df[dt_col] >= VAL_END].reset_index(drop=True)
    return train, val, test

# Test the functions
print("Loading and aggregating data...")
agg = build_aggregated()
train, val, test = split(agg)
print(f"Total hours: {len(agg)}  range: {agg.datetime.min()} -> {agg.datetime.max()}")
print(f"Train: {len(train)} hours ({train.datetime.min().date()} -> {train.datetime.max().date()})  ~{len(train)/730:.1f} mo")
print(f"Val:   {len(val)} hours ({val.datetime.min().date()} -> {val.datetime.max().date()})  ~{len(val)/730:.1f} mo")
print(f"Test:  {len(test)} hours ({test.datetime.min().date()} -> {test.datetime.max().date()})  ~{len(test)/730:.1f} mo")
print()
print("Real events already present in the data (no injection):")
print("Max SP15:", agg.loc[agg.SP15.idxmax(), ["datetime","SP15"]].to_dict())
print("Min SP15:", agg.loc[agg.SP15.idxmin(), ["datetime","SP15"]].to_dict())
agg.to_pickle("artifacts/agg_hourly.pkl")
print("\nSuccess! Saved aggregated data to artifacts/agg_hourly.pkl")
