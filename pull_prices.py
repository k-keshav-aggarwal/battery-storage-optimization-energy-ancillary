import time
import gridstatus
import pandas as pd
import numpy as np
import urllib3
import hashlib
from params import nodes, start_date, end_date, RANDOM_SEED

urllib3.disable_warnings()

iso = gridstatus.CAISO()



CACHE_LMP = "cache_lmp.pkl"
CACHE_AS  = "cache_as.pkl"


# =========================
# 🔹 LMP DATA (chunked + retry + cache)
# =========================
def get_lmp_data(start, end, nodes, chunk_days=15, sleep=12, retries=3):

    node_hash = hashlib.md5("".join(nodes).encode()).hexdigest()[:6]
    start_iso = pd.Timestamp(start).strftime("%Y%m%d")
    end_iso = pd.Timestamp(end).strftime("%Y%m%d")
    cache_key = f"cache_lmp_{start_iso}_{end_iso}_{node_hash}.pkl"

    if pd.io.common.file_exists(cache_key):
        print(f"[LMP] Loading from cache: {cache_key}")
        return pd.read_pickle(cache_key)

    chunks = pd.date_range(start, end, freq=f"{chunk_days}D")
    if chunks[-1] < pd.Timestamp(end):
        chunks = chunks.union(pd.DatetimeIndex([pd.Timestamp(end)]))

    frames = []

    for i in range(len(chunks) - 1):
        s, e_chunk = chunks[i], chunks[i + 1]
        success = False

        for attempt in range(1, retries + 1):
            try:
                df = iso.get_lmp(
                    start=s, end=e_chunk,
                    market="DAY_AHEAD_HOURLY",
                    locations=nodes,
                    sleep=sleep,
                )
                frames.append(df)
                print(f"[LMP] ✓ {s.date()} → {e_chunk.date()}")
                success = True
                break

            except Exception as ex:
                print(f"[LMP] attempt {attempt}/{retries} failed ({s.date()}→{e_chunk.date()}): {ex}")
                time.sleep(sleep * attempt)

        if not success:
            print(f"[LMP] ✗ Skipping chunk {s.date()} → {e_chunk.date()} after {retries} retries")

        time.sleep(sleep)

    if not frames:
        raise RuntimeError("LMP fetch failed for all chunks")

    lmp = pd.concat(frames, ignore_index=True)
    lmp = lmp.drop_duplicates()

    lmp["Time"] = pd.to_datetime(lmp["Time"], utc=True)
    lmp["datetime"] = lmp["Time"].dt.tz_convert(None)
    lmp = lmp[["datetime", "Location", "LMP"]]
    lmp.rename(columns={"Location": "node", "LMP": "SP15"}, inplace=True)

    lmp.to_pickle(cache_key)
    print(f"[LMP] Cached to {cache_key}")

    return lmp


# =========================
# 🔹 AS PRICES (retry + cache)
# =========================
def get_as_prices_data(start_date, end_date, chunk_days=15, sleep=12, retries=3):

    start_iso = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_iso = pd.Timestamp(end_date).strftime("%Y%m%d")
    cache_key = f"cache_as_{start_iso}_{end_iso}.pkl"

    if pd.io.common.file_exists(cache_key):
        print(f"[AS] Loading from cache: {cache_key}")
        return pd.read_pickle(cache_key)

    chunks = pd.date_range(start_date, end_date, freq=f"{chunk_days}D")
    if chunks[-1] < pd.Timestamp(end_date):
        chunks = chunks.union(pd.DatetimeIndex([pd.Timestamp(end_date)]))

    frames = []

    for i in range(len(chunks) - 1):
        s, e_chunk = chunks[i], chunks[i + 1]
        success = False

        for attempt in range(1, retries + 1):
            try:
                df = iso.get_as_prices(date=s, end=e_chunk)
                frames.append(df)
                print(f"[AS] ✓ {s.date()} → {e_chunk.date()}")
                success = True
                break

            except Exception as ex:
                print(f"[AS] attempt {attempt}/{retries} failed ({s.date()}→{e_chunk.date()}): {ex}")
                time.sleep(sleep * attempt)

        if not success:
            print(f"[AS] ✗ Skipping chunk {s.date()} → {e_chunk.date()} after {retries} retries")

        time.sleep(sleep)

    if not frames:
        raise RuntimeError("AS prices fetch failed for all chunks")

    as_prices = pd.concat(frames, ignore_index=True)
    as_prices = as_prices.drop_duplicates()

    as_prices["Time"] = pd.to_datetime(as_prices["Time"], utc=True)
    as_prices["datetime"] = as_prices["Time"].dt.tz_convert(None)

    numeric_cols = as_prices.select_dtypes(include=["number"]).columns.tolist()
    as_prices = as_prices[["datetime"] + numeric_cols]
    as_prices = as_prices.groupby("datetime").sum().reset_index()

    rename_map = {
        "Non-Spinning Reserves": "NonSpin",
        "Regulation Down":       "RegDown",
        "Regulation Up":         "RegUp",
        "Spinning Reserves":     "Spin",
    }

    missing = set(rename_map) - set(as_prices.columns)
    if missing:
        raise KeyError(f"AS price columns missing from API: {missing}")

    as_prices.rename(columns=rename_map, inplace=True)

    as_prices.to_pickle(cache_key)
    print(f"[AS] Cached to {cache_key}")

    return as_prices


# =========================
# 🔹 MERGE
# =========================
def get_merged_data():
    lmp      = get_lmp_data(start_date, end_date, nodes)
    as_prices = get_as_prices_data(start_date, end_date)

    if lmp is None or len(lmp) == 0:
        raise ValueError("LMP data is empty")
    if as_prices is None or len(as_prices) == 0:
        raise ValueError("AS prices data is empty")

    # Report coverage before merging
    print(f"[MERGE] LMP range:      {lmp['datetime'].min()} → {lmp['datetime'].max()}")
    print(f"[MERGE] AS range:       {as_prices['datetime'].min()} → {as_prices['datetime'].max()}")

    df = pd.merge(lmp, as_prices, on="datetime", how="inner")

    if len(df) == 0:
        raise ValueError("Merged dataframe is empty — check datetime alignment between LMP and AS")

    df = df.sort_values(["datetime", "node"]).reset_index(drop=True)
    print(f"[MERGE] Final shape: {df.shape}")

    return df


def inject_price_spike(df, magnitude=50, duration=6, seed=RANDOM_SEED):
    df = df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, max(1, len(df) - duration))
    df.iloc[idx:idx + duration, df.columns.get_loc("SP15")] += magnitude
    return df


# =========================
# 🔹 FINAL DATASETS
# =========================
merged_df_clean = get_merged_data()

if merged_df_clean is None or len(merged_df_clean) == 0:
    raise ValueError("merged_df_clean is empty or None")

merged_df_spike = inject_price_spike(merged_df_clean)