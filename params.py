# =========================
# SYSTEM PARAMETERS CONFIG
# =========================

# ---- Market Configuration ----
# MUST be multiple nodes for graph learning
nodes = [
    "TH_SP15_GEN-APND",
    "TH_NP15_GEN-APND",
    "TH_ZP26_GEN-APND"
]

products = ["SP15", "RegUp", "Spin", "RegDown", "NonSpin"]

# ---- Time Range ----
start_date = "Jan 1, 2023"
end_date = "Jan 31, 2023"

# ---- Battery Parameters ----
battery_capacity_mwh = 10.0
max_charge_power_mw = 10.0
max_discharge_power_mw = 10.0

mcp = battery_capacity_mwh
mdp = max_discharge_power_mw

# ---- Efficiency ----
efficiency = 0.80
e = efficiency

# ---- Market Fees ----
transaction_fee = 1.0
fee = transaction_fee

RANDOM_SEED = 42