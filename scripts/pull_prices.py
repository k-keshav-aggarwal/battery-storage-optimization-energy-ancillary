"""Fetch, cache, and combine market price data for the notebook.

The helpers below load LMP and ancillary-service data, align them on a common
timestamp, and create a reproducible spike scenario for the attack dataset.
"""

import argparse
import io
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np
import hashlib
from params import nodes, start_date, end_date, RANDOM_SEED

try:
    import gridstatus
except ImportError:
    gridstatus = None

CACHE_LMP = "cache_lmp.pkl"
CACHE_AS  = "cache_as.pkl"
OASIS_URL = "https://oasis.caiso.com/oasisapi/SingleZip"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = REPO_ROOT / "data_cache"


def _oasis_csv(params):
    """Fetch and unpack one OASIS CSV response."""
    query = urlencode(params)
    request = Request(f"{OASIS_URL}?{query}", headers={"User-Agent": "CAISO-research-fetch/1.0"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.namelist()
        csv_members = [name for name in members if name.lower().endswith(".csv")]
        if not csv_members:
            xml = archive.read(members[0]).decode("utf-8", errors="replace")
            raise RuntimeError(f"OASIS returned no CSV data: {xml[:500]}")
        return pd.read_csv(archive.open(csv_members[0]))


def _api_date(value):
    """Format a CAISO day boundary in the API's required UTC notation."""
    return pd.Timestamp(value).strftime("%Y%m%dT08:00-0000")


def _chunks(start, end, days=31):
    current = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while current < end:
        next_date = min(current + pd.Timedelta(days=days), end)
        yield current, next_date
        current = next_date


def _expected_hours(start, end):
    return pd.date_range(start, end, inclusive="left", freq="h", tz="UTC").tz_localize(None)


# LMP data: chunked fetch with retry and cache.
def get_lmp_data(start, end, nodes, chunk_days=31, sleep=3, retries=3,
                 cache_dir=DATA_CACHE, refresh=False):
    """Fetch day-ahead LMP data for the selected nodes.

    The request is split into chunks so a long time range can be retried in
    smaller pieces and cached locally for repeat runs.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    node_hash = hashlib.md5("".join(nodes).encode()).hexdigest()[:6]
    start_iso = pd.Timestamp(start).strftime("%Y%m%d")
    end_iso = pd.Timestamp(end).strftime("%Y%m%d")
    cache_key = cache_dir / f"cache_lmp_{start_iso}_{end_iso}_{node_hash}.pkl"

    if cache_key.exists() and not refresh:
        print(f"[LMP] Loading from cache: {cache_key}")
        return pd.read_pickle(cache_key)

    frames = []

    for s, e_chunk in _chunks(start, end, chunk_days):
        success = False

        for attempt in range(1, retries + 1):
            try:
                df = _oasis_csv({
                    "resultformat": "6",
                    "queryname": "PRC_LMP",
                    "version": "12",
                    "market_run_id": "DAM",
                    "node": ",".join(nodes),
                    "startdatetime": _api_date(s),
                    "enddatetime": _api_date(e_chunk),
                })
                frames.append(df)
                print(f"[LMP] OK {s.date()} -> {e_chunk.date()}")
                success = True
                break

            except Exception as ex:
                print(f"[LMP] attempt {attempt}/{retries} failed ({s.date()}->{e_chunk.date()}): {ex}")
                time.sleep(sleep * attempt)

        if not success:
            print(f"[LMP] Skipping chunk {s.date()} -> {e_chunk.date()} after {retries} retries")

        time.sleep(sleep)

    if not frames:
        raise RuntimeError("LMP fetch failed for all chunks")

    lmp = pd.concat(frames, ignore_index=True)
    lmp = lmp.drop_duplicates()

    lmp["datetime"] = pd.to_datetime(lmp["INTERVALSTARTTIME_GMT"], utc=True).dt.tz_localize(None)
    lmp = lmp[["datetime", "NODE", "MW"]].rename(columns={"NODE": "node", "MW": "SP15"})
    lmp["node"] = lmp["node"].astype(str)
    lmp["SP15"] = pd.to_numeric(lmp["SP15"], errors="coerce")
    lmp = lmp.dropna(subset=["datetime", "SP15"]).drop_duplicates(
        subset=["datetime", "node"]
    ).sort_values(["datetime", "node"]).reset_index(drop=True)

    lmp.to_pickle(cache_key)
    print(f"[LMP] Cached to {cache_key}")

    return lmp


def refresh_retrievable_lmp(start="2023-04-01", end="2026-01-01"):
    """Refresh 2024-2025 from OASIS and retain the archived 2023 cache.

    OASIS no longer serves the January-December 2023 interval.  The existing
    cache is therefore used only for 2023, while 2024-2025 are fetched live.
    """
    old_path = REPO_ROOT / "cache_lmp_20230101_20251231_0c3247.pkl"
    old = pd.read_pickle(old_path)
    old["datetime"] = pd.to_datetime(old["datetime"])
    archived = old[old["datetime"] >= pd.Timestamp(start)].copy()
    live = get_lmp_data("2024-01-01", end, nodes, refresh=True)
    # Prefer fresh rows and use the archived cache only for OASIS gaps/rate
    # limits.  No synthetic values are introduced.
    combined = pd.concat([live, archived], ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime", "node"], keep="first")
    combined = combined.sort_values(["datetime", "node"]).reset_index(drop=True)
    expected = set(archived["datetime"].unique())
    actual = set(combined["datetime"].unique())
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(
            f"Combined LMP data has {len(missing)} missing UTC hours; "
            f"first missing hours: {missing[:5]}"
        )
    end_label = pd.Timestamp(end).strftime("%Y%m%d")
    start_label = pd.Timestamp(start).strftime("%Y%m%d")
    output = DATA_CACHE / f"lmp_combined_{start_label}_{end_label}.pkl"
    combined.to_pickle(output)
    print(f"[LMP] Refreshed combined cache: {output} ({len(combined)} rows)")
    return combined


def write_canonical_cache(lmp):
    """Align LMP with the archived ancillary-service data and write CSV cache."""
    as_path = REPO_ROOT / "cache_as_20230101_20251231.pkl"
    as_prices = pd.read_pickle(as_path)
    as_prices["datetime"] = pd.to_datetime(as_prices["datetime"])
    merged = lmp.merge(as_prices, on="datetime", how="inner")
    merged = merged.sort_values(["datetime", "node"]).reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between LMP and ancillary data")
    counts = merged.groupby("datetime")["node"].nunique()
    incomplete = counts[counts != len(nodes)]
    if not incomplete.empty:
        raise RuntimeError(
            f"Canonical cache has {len(incomplete)} hours without all {len(nodes)} hubs; "
            f"first incomplete hour: {incomplete.index[0]}"
        )
    output = DATA_CACHE / "merged_df_clean.csv.gz"
    merged.to_csv(output, index=False, compression="gzip")
    print(
        f"[MERGE] Wrote {output}: {len(merged)} rows, "
        f"{merged['datetime'].nunique()} complete UTC hours"
    )
    return merged


# AS prices: chunked fetch with retry and cache.
def get_as_prices_data(start_date, end_date, chunk_days=15, sleep=12, retries=3):
    """Fetch ancillary-service prices and standardise the column names.

    The returned table is grouped by timestamp so it can be merged with the LMP
    data on the notebook's shared datetime column.
    """

    start_iso = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_iso = pd.Timestamp(end_date).strftime("%Y%m%d")
    cache_key = f"cache_as_{start_iso}_{end_iso}.pkl"

    if pd.io.common.file_exists(cache_key):
        print(f"[AS] Loading from cache: {cache_key}")
        return pd.read_pickle(cache_key)
    if gridstatus is None:
        raise RuntimeError(
            "Ancillary-service cache is unavailable and gridstatus is not installed. "
            "Install gridstatus or provide a validated ancillary-service cache."
        )
    iso = gridstatus.CAISO()

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


# Merge the LMP and AS datasets on timestamp.
def get_merged_data():
    """Load both price datasets and join them into one analysis table."""
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
    """Create the attack scenario by adding a deterministic price spike."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, max(1, len(df) - duration))
    df.iloc[idx:idx + duration, df.columns.get_loc("SP15")] += magnitude
    return df


def main():
    parser = argparse.ArgumentParser(description="Refresh and validate CAISO research data")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="fetch currently retrievable LMP data from OASIS and rebuild the canonical cache",
    )
    parser.add_argument(
        "--start",
        default="2023-04-01",
        help="inclusive UTC start date for the final cache (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2026-01-01",
        help="exclusive UTC end date for the live LMP refresh (YYYY-MM-DD)",
    )
    args = parser.parse_args()
    if not args.refresh:
        raise SystemExit("Use --refresh to explicitly fetch live CAISO data.")
    lmp = refresh_retrievable_lmp(args.start, args.end)
    merged_df_clean = write_canonical_cache(lmp)
    inject_price_spike(merged_df_clean).to_csv(
        DATA_CACHE / "merged_df_spike.csv.gz", index=False, compression="gzip"
    )


if __name__ == "__main__":
    main()