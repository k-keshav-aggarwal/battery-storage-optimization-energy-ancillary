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

mcp = battery_capacity_mwh
mdp = max_discharge_power_mw

# ---- Efficiency & Costs ----
efficiency = 0.80
e = efficiency

transaction_fee = 1.0
fee = transaction_fee

# [IMPROVEMENT]: Added degradation cost per MWh of throughput
degradation_cost = 2.5 

RANDOM_SEED = 42