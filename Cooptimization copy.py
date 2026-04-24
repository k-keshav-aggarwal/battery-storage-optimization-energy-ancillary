import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from pull_prices import (
    merged_df_clean,
    merged_df_spike,
    merged_df_iqr,
    merged_df_median,
    merged_df_zscore,
)

from params import (
    mcp,
    mdp,
    e,
    fee,
    nodes,
    products,
    RANDOM_SEED,
)

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(RANDOM_SEED)

# -------------------------
# Output Directory
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "optimization_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Solver
# -------------------------
def get_available_solver():
    for solver_name in ["highs", "appsi_highs", "glpk", "cbc"]:
        print(f"Trying solver: {solver_name}")
        try:
            solver = pyo.SolverFactory(solver_name)
            if solver and solver.available(exception_flag=False):
                print(f"Using solver: {solver_name}")
                return solver_name, solver
        except Exception as e:
            print(f"{solver_name} failed: {e}")
    raise RuntimeError("No solver available.")


def get_solver_options(name):
    if name in ["highs", "appsi_highs"]:
        return {"time_limit": 300, "mip_rel_gap": 0.05}
    if name == "glpk":
        return {"tmlim": 300, "mipgap": 0.05}
    return {"seconds": 300, "ratio": 0.05}


# -------------------------
# Optimization Model
# -------------------------
def build_and_solve_model(merged_df, label):
    merged_df = merged_df.copy()
    merged_df["datetime"] = pd.to_datetime(merged_df["datetime"])

    model = pyo.ConcreteModel()

    model.t = pyo.Set(initialize=range(1, len(merged_df) + 1))
    model.p = pyo.Set(initialize=products)
    model.n = pyo.Set(initialize=nodes)

    model.price = pyo.Param(model.t, model.p, model.n, mutable=True)

    for t in model.t:
        for p in model.p:
            for n in model.n:
                model.price[t, p, n] = float(merged_df.loc[t - 1, p])

    model.is_buy = pyo.Var(model.t, model.n, within=pyo.Binary)
    model.buy = pyo.Var(model.t, model.p, model.n, bounds=(0, None))
    model.sell = pyo.Var(model.t, model.p, model.n, bounds=(0, None))

    model.soc = pyo.Var(model.t, bounds=(0, mcp))
    model.net_energy_flow = pyo.Var(model.t, bounds=(-mdp, mcp))

    def battery_dynamics(m, t):
        if t == 1:
            return m.soc[t] == mcp + m.net_energy_flow[t]
        return m.soc[t] == m.soc[t - 1] + m.net_energy_flow[t]

    model.battery = pyo.Constraint(model.t, rule=battery_dynamics)

    def net_flow(m, t):
        bought = sum(m.buy[t, "SP15", n] for n in m.n)
        sold = sum(m.sell[t, "SP15", n] for n in m.n)
        return m.net_energy_flow[t] == bought * e - sold / e

    model.net_flow = pyo.Constraint(model.t, rule=net_flow)

    def buy_limit(m, t, p, n):
        return m.buy[t, p, n] <= mcp * m.is_buy[t, n]

    def sell_limit(m, t, p, n):
        return m.sell[t, p, n] <= mdp * (1 - m.is_buy[t, n])

    model.buy_constr = pyo.Constraint(model.t, model.p, model.n, rule=buy_limit)
    model.sell_constr = pyo.Constraint(model.t, model.p, model.n, rule=sell_limit)

    model.charge = pyo.Constraint(
        model.t, model.n,
        rule=lambda m, t, n: m.buy[t, "SP15", n] <= (mcp - m.soc[t]) / e
    )

    model.discharge = pyo.Constraint(
        model.t, model.n,
        rule=lambda m, t, n: m.sell[t, "SP15", n] <= m.soc[t] * e
    )

    def as_limit(m, t, n):
        return sum(
            m.buy[t, p, n] + m.sell[t, p, n]
            for p in ["RegUp", "Spin", "RegDown", "NonSpin"]
        ) <= mcp

    model.as_limit = pyo.Constraint(model.t, model.n, rule=as_limit)

    def objective(m):
        return sum(
            m.sell[t, p, n] * (m.price[t, p, n] - fee)
            - m.buy[t, p, n] * (m.price[t, p, n] + fee)
            for t in m.t for p in m.p for n in m.n
        )

    model.obj = pyo.Objective(rule=objective, sense=pyo.maximize)

    solver_name, solver = get_available_solver()
    solver.solve(model, tee=False, options=get_solver_options(solver_name))

    # Extract data
    battery = [pyo.value(model.soc[t]) for t in model.t]
    pnl = []

    for t in model.t:
        val = 0
        for p in model.p:
            for n in model.n:
                price = model.price[t, p, n]
                val += (
                    pyo.value(model.sell[t, p, n]) * (price - fee)
                    - pyo.value(model.buy[t, p, n]) * (price + fee)
                )
        pnl.append(val)

    pnl = np.array(pnl)

    return {
        "label": label,
        "prices": merged_df,
        "battery": np.array(battery),
        "pnl": pnl,
        "summary": {
            "case": label,
            "objective_profit": float(pyo.value(model.obj)),
            "price_volatility_mad": float(np.mean(np.abs(merged_df["SP15"] - merged_df["SP15"].mean()))),
        }
    }


# -------------------------
# Metrics
# -------------------------
def add_comparison_metrics(outputs):
    clean_soc = outputs["clean"]["battery"]
    clean_profit = outputs["clean"]["summary"]["objective_profit"]

    for k, v in outputs.items():
        v["summary"]["soc_dev"] = float(np.mean(np.abs(v["battery"] - clean_soc)))
        v["summary"]["profit_delta"] = v["summary"]["objective_profit"] - clean_profit

    return outputs


def percent_recovery(a, m, c):
    d = a - c
    return 0 if abs(d) < 1e-12 else (a - m) / d * 100


def make_summary_table(outputs):
    rows = [outputs[k]["summary"] for k in outputs]
    df = pd.DataFrame(rows)

    clean = df[df.case == "clean"].iloc[0]
    attack = df[df.case == "attack"].iloc[0]

    rec = []
    for case in ["zscore", "median", "iqr"]:
        row = df[df.case == case].iloc[0]
        rec.append({
            "case": case,
            "profit_recovery": percent_recovery(
                attack.objective_profit,
                row.objective_profit,
                clean.objective_profit
            )
        })

    return df, pd.DataFrame(rec)


# -------------------------
# Plots
# -------------------------
def save_plots(outputs):
    labels = ["clean", "attack", "zscore", "median", "iqr"]

    profits = [outputs[k]["summary"]["objective_profit"] for k in labels]

    plt.figure()
    plt.bar(labels, profits)
    plt.title("Profit Comparison")
    plt.savefig(os.path.join(OUTPUT_DIR, "profit.png"))
    plt.close()


# -------------------------
# MAIN
# -------------------------
def main():
    outputs = {
        "clean": build_and_solve_model(merged_df_clean, "clean"),
        "attack": build_and_solve_model(merged_df_spike, "attack"),
        "zscore": build_and_solve_model(merged_df_zscore, "zscore"),
        "median": build_and_solve_model(merged_df_median, "median"),
        "iqr": build_and_solve_model(merged_df_iqr, "iqr"),
    }

    outputs = add_comparison_metrics(outputs)

    summary, recovery = make_summary_table(outputs)

    print(summary)
    print(recovery)

    save_plots(outputs)

if __name__ == "__main__":
    main()


# -------------------------
# EXTRA UTILITIES
# -------------------------
def run_single_case(case="clean"):
    mapping = {
        "clean": merged_df_clean,
        "attack": merged_df_spike,
        "zscore": merged_df_zscore,
        "median": merged_df_median,
        "iqr": merged_df_iqr,
    }

    if case not in mapping:
        print("Invalid case")
        return

    result = build_and_solve_model(mapping[case], case)
    print(result["summary"])