"""Shared configuration values for the battery dispatch notebook.

The notebook imports these constants so the market setup, battery limits,
efficiency assumptions, and random seed stay in one place.
"""

# =========================
# SYSTEM PARAMETERS CONFIG
# =========================

# ---- Market Configuration ----
nodes = [
    "TH_SP15_GEN-APND",
    "TH_NP15_GEN-APND",
    "TH_ZP26_GEN-APND"
]

products = ["SP15", "RegUp", "Spin", "RegDown", "NonSpin"]

# ---- Time Range ----
start_date = "Jan 1, 2023"
end_date = "Dec 31, 2025"

# ---- Battery Parameters ----
battery_capacity_mwh = 10.0
max_charge_power_mw = 10.0
max_discharge_power_mw = 10.0

# Notebook-compatible aliases.
mcp = battery_capacity_mwh
mdp = max_discharge_power_mw

# ---- Efficiency & Costs ----
round_trip_efficiency = 0.80
efficiency = round_trip_efficiency
e = efficiency

transaction_fee = 1.0
fee = transaction_fee

# Throughput-based degradation cost per MWh.
degradation_cost = 2.5 

RANDOM_SEED = 42