#!/usr/bin/env python3
"""
scripts/pull_october_data.py
============================
Fetch October 2026 CAISO LMP and Ancillary Service prices up to the current
available market horizon, and integrate them into data_cache.
"""

import sys
import time
import socket
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure DNS resilience for CAISO OASIS endpoint
_orig_getaddrinfo = socket.getaddrinfo

def _resilient_getaddrinfo(host, port, *args, **kwargs):
    if host in ("oasis.caiso.com", "oasis.prod.caiso.com"):
        try:
            return _orig_getaddrinfo(host, port, *args, **kwargs)
        except Exception:
            return _orig_getaddrinfo("45.42.22.33", port, *args, **kwargs)
    return _orig_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = _resilient_getaddrinfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pull_prices import _oasis_csv, _api_date, nodes, DATA_CACHE, inject_price_spike


def fetch_october_lmp(start="2026-10-01", end="2026-10-11"):
    out_live = DATA_CACHE / "cache_lmp_20240101_20261011_0c3247.pkl"
    if out_live.exists():
        print(f"[LMP] Found existing October live cache: {out_live.name}")
        return pd.read_pickle(out_live)

    print(f"[LMP] Fetching October data: {start} -> {end}...")
    time.sleep(2)
    df = _oasis_csv({
        "resultformat": "6",
        "queryname": "PRC_LMP",
        "version": "12",
        "market_run_id": "DAM",
        "node": ",".join(nodes),
        "startdatetime": _api_date(start),
        "enddatetime": _api_date(end),
    })

    df["datetime"] = pd.to_datetime(df["INTERVALSTARTTIME_GMT"], utc=True).dt.tz_localize(None)
    df = df[["datetime", "NODE", "MW"]].rename(columns={"NODE": "node", "MW": "SP15"})
    df["node"] = df["node"].astype(str)
    df["SP15"] = pd.to_numeric(df["SP15"], errors="coerce")
    df = df.dropna(subset=["datetime", "SP15"]).drop_duplicates(
        subset=["datetime", "node"]
    ).sort_values(["datetime", "node"]).reset_index(drop=True)

    print(f"[LMP] Fetched {len(df)} rows across {df['node'].nunique()} nodes.")
    print(f"      Coverage: {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def update_lmp_caches(df_oct, end_label="20261011"):
    out_comb = DATA_CACHE / f"lmp_combined_20230112_{end_label}.pkl"
    if out_comb.exists():
        print(f"[LMP] Combined cache already present: {out_comb.name}")
        return pd.read_pickle(out_comb)

    cache_2024 = DATA_CACHE / "cache_lmp_20240101_20261001_0c3247.pkl"
    if cache_2024.exists():
        prev_live = pd.read_pickle(cache_2024)
        live_combined = pd.concat([prev_live, df_oct], ignore_index=True)
        live_combined = live_combined.drop_duplicates(subset=["datetime", "node"], keep="last")
        live_combined = live_combined.sort_values(["datetime", "node"]).reset_index(drop=True)
    else:
        live_combined = df_oct

    out_live = DATA_CACHE / f"cache_lmp_20240101_{end_label}_0c3247.pkl"
    live_combined.to_pickle(out_live)
    print(f"[LMP] Saved live cache: {out_live.name} ({len(live_combined):,} rows, max: {live_combined['datetime'].max()})")

    old_raw = pd.read_pickle(REPO_ROOT / "cache_lmp_20230101_20251231_0c3247.pkl")
    old_raw["datetime"] = pd.to_datetime(old_raw["datetime"])

    for start_date, start_lbl in [("2023-04-01", "20230401"), ("2023-01-12", "20230112")]:
        archived = old_raw[old_raw["datetime"] >= pd.Timestamp(start_date)].copy()
        combined = pd.concat([live_combined, archived], ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime", "node"], keep="first")
        combined = combined.sort_values(["datetime", "node"]).reset_index(drop=True)

        target_file = DATA_CACHE / f"lmp_combined_{start_lbl}_{end_label}.pkl"
        combined.to_pickle(target_file)
        print(f"[LMP] Saved combined cache: {target_file.name} ({len(combined):,} rows, {combined['datetime'].min()} to {combined['datetime'].max()})")

    return live_combined


def fetch_as_prices(start="2026-01-01", end="2026-10-11", chunk_days=14):
    print(f"\n[AS] Fetching Ancillary Services data: {start} -> {end} in {chunk_days}-day chunks...")
    chk_file = DATA_CACHE / "as_2026_checkpoint.pkl"
    if chk_file.exists():
        as_cached = pd.read_pickle(chk_file)
        print(f"     [AS] Loaded checkpoint with {len(as_cached)} already fetched hours (up to {as_cached['datetime'].max()})")
    else:
        as_cached = pd.DataFrame()

    chunks = pd.date_range(start, end, freq=f"{chunk_days}D")
    if chunks[-1] < pd.Timestamp(end):
        chunks = chunks.union(pd.DatetimeIndex([pd.Timestamp(end)]))

    frames = [as_cached] if not as_cached.empty else []
    already_covered = set(pd.to_datetime(as_cached["datetime"])) if not as_cached.empty else set()

    rename_map = {
        "NR": "NonSpin",
        "RD": "RegDown",
        "RU": "RegUp",
        "SR": "Spin",
        "RMD": "Regulation Mileage Down",
        "RMU": "Regulation Mileage Up",
    }

    for i in range(len(chunks) - 1):
        s, e_chunk = chunks[i], chunks[i + 1]
        chunk_hours = pd.date_range(s, e_chunk, freq="h", inclusive="left")
        if chunk_hours.isin(already_covered).all():
            print(f"     [AS] Skipping cached chunk: {s.date()} -> {e_chunk.date()}")
            continue

        success = False
        for attempt in range(1, 4):
            try:
                time.sleep(2.5)
                df = _oasis_csv({
                    "resultformat": "6",
                    "queryname": "PRC_AS",
                    "version": "1",
                    "market_run_id": "DAM",
                    "startdatetime": _api_date(s),
                    "enddatetime": _api_date(e_chunk),
                })
                df["datetime"] = pd.to_datetime(df["INTERVALSTARTTIME_GMT"], utc=True).dt.tz_localize(None)
                df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
                summed = df.groupby(["datetime", "ANC_TYPE"])["MW"].sum().unstack()
                summed = summed.rename(columns=rename_map).reset_index()
                frames.append(summed)
                print(f"     [AS] OK {s.date()} -> {e_chunk.date()} ({len(summed)} hours)")
                success = True
                
                # Checkpoint
                current_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
                current_all.to_pickle(chk_file)
                break
            except Exception as ex:
                print(f"     [AS] Retry {attempt} ({s.date()} -> {e_chunk.date()}): {ex}")
                time.sleep(4 * attempt)

        if not success:
            raise RuntimeError(f"Failed to fetch AS prices for chunk {s} -> {e_chunk}")

    as_df = pd.concat(frames, ignore_index=True)
    as_df = as_df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    cols = ["datetime", "NonSpin", "RegDown", "Regulation Mileage Down", "Regulation Mileage Up", "RegUp", "Spin"]
    for c in cols:
        if c not in as_df.columns:
            as_df[c] = 0.0
    as_df = as_df[cols]

    print(f"[AS] Total 2026 AS collected: {len(as_df):,} hours from {as_df['datetime'].min()} to {as_df['datetime'].max()}")
    return as_df


def update_as_cache(as_2026, end_label="20261011"):
    as_old_path = REPO_ROOT / "cache_as_20230101_20251231.pkl"
    as_old = pd.read_pickle(as_old_path)
    as_old["datetime"] = pd.to_datetime(as_old["datetime"])

    as_combined = pd.concat([as_old, as_2026], ignore_index=True)
    as_combined = as_combined.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)

    out_as = DATA_CACHE / f"cache_as_20230101_{end_label}.pkl"
    as_combined.to_pickle(out_as)
    print(f"[AS] Saved combined AS cache: {out_as.name} ({len(as_combined):,} rows, max: {as_combined['datetime'].max()})")
    return as_combined


def build_merged_october_data(lmp_df, as_df):
    print("\n[MERGE] Merging updated LMP and Ancillary Services through October 2026...")
    merged = lmp_df.merge(as_df, on="datetime", how="inner")
    merged = merged.sort_values(["datetime", "node"]).reset_index(drop=True)

    out_clean = DATA_CACHE / "merged_df_clean.csv.gz"
    merged.to_csv(out_clean, index=False, compression="gzip")
    print(f"[MERGE] Updated canonical cache: {out_clean.name} ({len(merged):,} rows across {merged['datetime'].nunique()} hours)")
    print(f"        Coverage: {merged['datetime'].min()} to {merged['datetime'].max()}")

    # Generate corresponding spike dataset
    out_spike = DATA_CACHE / "merged_df_spike.csv.gz"
    inject_price_spike(merged).to_csv(out_spike, index=False, compression="gzip")
    print(f"[MERGE] Generated corresponding spike dataset: {out_spike.name}")
    return merged


def main():
    print("=" * 70)
    print(" CAISO OASIS LIVE PULL: EXTENDING HORIZON THROUGH OCTOBER 2026")
    print("=" * 70)

    # 1. Pull October LMP (2026-10-01 to 2026-10-11)
    df_oct_lmp = fetch_october_lmp("2026-10-01", "2026-10-11")
    lmp_all = update_lmp_caches(df_oct_lmp, "20261011")

    # 2. Pull 2026 Ancillary Services through October 11, 2026
    as_2026 = fetch_as_prices("2026-01-01", "2026-10-11", chunk_days=14)
    as_all = update_as_cache(as_2026, "20261011")

    # 3. Create merged dataset including October 2026
    lmp_full = pd.read_pickle(DATA_CACHE / "lmp_combined_20230112_20261011.pkl")
    build_merged_october_data(lmp_full, as_all)

    # Clean up checkpoint
    chk_file = DATA_CACHE / "as_2026_checkpoint.pkl"
    if chk_file.exists():
        chk_file.unlink()

    print("\n" + "=" * 70)
    print(" SUCCESS: OCTOBER 2026 DATA FULLY PULLED AND INTEGRATED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
