"""Central configuration: battery physics, market setup, splits, seeds.

Every constant the pipeline uses lives here so a run is fully described by
this file plus a seed. Nothing downstream hardcodes a number.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------- battery ---
# One-way efficiency. Round-trip is EFFICIENCY**2 -- stated explicitly here
# because the original params.py called the one-way figure "round_trip".
EFFICIENCY_ONE_WAY = 0.90
EFFICIENCY_ROUND_TRIP = EFFICIENCY_ONE_WAY ** 2  # 0.81

CAPACITY_MWH = 10.0
MAX_CHARGE_MW = 10.0
MAX_DISCHARGE_MW = 10.0
INITIAL_SOC_FRAC = 0.5

TRANSACTION_FEE = 0.5      # $/MWh, charged on both legs
DEGRADATION_COST = 2.5     # $/MWh of throughput

# ------------------------------------------------------------------ market ---
NODES = ["TH_SP15_GEN-APND", "TH_NP15_GEN-APND", "TH_ZP26_GEN-APND"]

# CAISO day-ahead market clears once per day for the following 24 hours.
# The dispatcher therefore optimises a 24-hour block against the announced
# schedule -- this is the real market structure, not perfect foresight.
DA_HORIZON_HOURS = 24

# ------------------------------------------------------------------ splits ---
# Chronological, aligned to calendar boundaries so cutoffs are auditable.
TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-07-01")

# ------------------------------------------------------------------- seeds ---
SEEDS = [0, 1, 2, 3, 4]

# --------------------------------------------------------------- detection ---
VAE_SEQ_LEN = 24
VAE_LATENT_DIM = 8
VAE_EPOCHS = 40
VAE_BATCH = 64
VAE_LR = 1e-3

# Ensemble weights (VAE, iForest, LOF) -- unchanged from the original work.
ENSEMBLE_WEIGHTS = (0.50, 0.30, 0.20)

# Hours scoring above this are passed to the genuine-vs-synthetic classifier.
ANOMALY_THRESHOLD = 0.15

# ------------------------------------------------------------------ attack ---
# Injection rate rather than a fixed count, so every split receives a
# proportional number of events and the test window is not starved.
SPIKE_INTERVAL_HOURS = 300     # ~1 event per 300 hours
SPIKE_MAGNITUDE = 3.0
SPIKE_DURATION_HOURS = 8
NON_REVERT_FRACTION = 0.9

# ------------------------------------------------------------------- paths ---
CACHE_DIR = "data_cache"
ARTIFACT_DIR = "artifacts_v2"
FIG_DIR = "figs_v2"
