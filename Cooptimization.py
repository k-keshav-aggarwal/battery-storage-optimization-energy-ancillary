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
from params import mcp, mdp, e, fee, nodes, products


OUTPUT_DIR = os.path.join(os.getcwd(), "optimization_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_available_solver():
    for solver_name in ["highs", "appsi_highs", "glpk", "cbc"]:
        try:
            candidate = pyo.SolverFactory(solver_name)
            if candidate is not None and candidate.available(exception_flag=False):
                return solver_name, candidate
        except Exception:
            continue
    raise RuntimeError("No supported solver found. Install one of: highspy, GLPK, or CBC.")


def get_solver_options(selected_solver_name):
    if selected_solver_name in ["highs", "appsi_highs"]:
        return {"time_limit": 300, "mip_rel_gap": 0.05}
    if selected_solver_name == "glpk":
        return {"tmlim": 300, "mipgap": 0.05}
    return {"seconds": 300, "ratio": 0.05}


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

    model.is_buy = pyo.Var(model.t, model.n, within=pyo.Binary, initialize=0)

    model.buy = pyo.Var(model.t, model.p, model.n, bounds=(0, None), initialize=0)
    model.sell = pyo.Var(model.t, model.p, model.n, bounds=(0, None), initialize=0)

    model.soc = pyo.Var(model.t, bounds=(0, mcp), initialize=mcp)
    model.net_energy_flow = pyo.Var(model.t, bounds=(-mdp, mcp), initialize=0)

    def battery_dynamics(model, t):
        if t == 1:
            return model.soc[t] == mcp + model.net_energy_flow[t]
        return model.soc[t] == model.soc[t - 1] + model.net_energy_flow[t]

    model.battery_dynamics_constr = pyo.Constraint(model.t, rule=battery_dynamics)

    def soc_min(model, t):
        return model.soc[t] >= 0

    def soc_max(model, t):
        return model.soc[t] <= mcp

    model.soc_min_constr = pyo.Constraint(model.t, rule=soc_min)
    model.soc_max_constr = pyo.Constraint(model.t, rule=soc_max)

    def net_flow_calc(model, t, n):
        energy_bought = model.buy[t, "SP15", n]
        energy_sold = model.sell[t, "SP15", n]

        charge_contribution = energy_bought * e
        discharge_contribution = energy_sold / e

        return model.net_energy_flow[t] == charge_contribution - discharge_contribution

    model.net_flow_calc_constr = pyo.Constraint(model.t, model.n, rule=net_flow_calc)

    def buy_sell_exclusivity_buy(model, t, p, n):
        limit = mcp if p == "SP15" else mcp
        return model.buy[t, p, n] <= limit * model.is_buy[t, n]

    def buy_sell_exclusivity_sell(model, t, p, n):
        limit = mdp if p == "SP15" else mdp
        return model.sell[t, p, n] <= limit * (1 - model.is_buy[t, n])

    model.buy_exclusivity = pyo.Constraint(model.t, model.p, model.n, rule=buy_sell_exclusivity_buy)
    model.sell_exclusivity = pyo.Constraint(model.t, model.p, model.n, rule=buy_sell_exclusivity_sell)

    def charge_rate_limit(model, t, n):
        if t == 1:
            available_capacity = (mcp - mcp) / e
        else:
            available_capacity = (mcp - model.soc[t]) / e
        return model.buy[t, "SP15", n] <= available_capacity + mcp

    model.charge_rate_constr = pyo.Constraint(model.t, model.n, rule=charge_rate_limit)

    def discharge_rate_limit(model, t, n):
        available_energy = model.soc[t] * e
        return model.sell[t, "SP15", n] <= available_energy

    model.discharge_rate_constr = pyo.Constraint(model.t, model.n, rule=discharge_rate_limit)

    def as_total_limit(model, t, n):
        total_as_buy = sum(model.buy[t, p, n] for p in ["RegUp", "Spin", "RegDown", "NonSpin"])
        total_as_sell = sum(model.sell[t, p, n] for p in ["RegUp", "Spin", "RegDown", "NonSpin"])
        return total_as_buy + total_as_sell <= mcp

    model.as_total_limit_constr = pyo.Constraint(model.t, model.n, rule=as_total_limit)

    def objective(model):
        profit = 0
        for t in model.t:
            for p in model.p:
                for n in model.n:
                    sell_revenue = model.sell[t, p, n] * (model.price[t, p, n] - fee)
                    buy_cost = model.buy[t, p, n] * (model.price[t, p, n] + fee)
                    profit += sell_revenue - buy_cost
        return profit

    model.objective = pyo.Objective(rule=objective, sense=pyo.maximize)

    selected_solver_name, solver = get_available_solver()
    solver_options = get_solver_options(selected_solver_name)

    print(f"\nRunning case: {label}")
    print(f"Using solver: {selected_solver_name}")

    results = solver.solve(model, tee=False, options=solver_options)

    if results.solver.status != pyo.SolverStatus.ok:
        raise RuntimeError(
            f"{label}: solving failed | status={results.solver.status} | "
            f"termination={results.solver.termination_condition}"
        )

    results_data = []
    for t in model.t:
        for p in model.p:
            for n in model.n:
                results_data.append({
                    "case": label,
                    "time_step": t,
                    "datetime": merged_df.loc[t - 1, "datetime"],
                    "product": p,
                    "node": n,
                    "price": float(model.price[t, p, n].value),
                    "buy_qty": float(model.buy[t, p, n].value),
                    "sell_qty": float(model.sell[t, p, n].value),
                    "is_buy": float(model.is_buy[t, n].value) if p == products[0] else np.nan,
                })

    results_df = pd.DataFrame(results_data)

    battery_results = []
    for t in model.t:
        battery_results.append({
            "case": label,
            "time_step": t,
            "datetime": merged_df.loc[t - 1, "datetime"],
            "soc": float(model.soc[t].value),
            "net_flow": float(model.net_energy_flow[t].value),
        })

    battery_df = pd.DataFrame(battery_results)

    pnl_data = []
    for t in model.t:
        hourly_pnl = 0.0
        for p in model.p:
            for n in model.n:
                sell_pnl = float(model.sell[t, p, n].value) * (float(model.price[t, p, n].value) - fee)
                buy_pnl = -float(model.buy[t, p, n].value) * (float(model.price[t, p, n].value) + fee)
                hourly_pnl += sell_pnl + buy_pnl

        pnl_data.append({
            "case": label,
            "time_step": t,
            "datetime": merged_df.loc[t - 1, "datetime"],
            "hourly_pnl": hourly_pnl,
        })

    pnl_df = pd.DataFrame(pnl_data)
    pnl_df["cumulative_pnl"] = pnl_df["hourly_pnl"].cumsum()

    summary = {
        "case": label,
        "objective_profit": float(pyo.value(model.objective)),
        "price_volatility_mad": float(np.mean(np.abs(merged_df["SP15"] - merged_df["SP15"].mean()))),
        "avg_soc": float(battery_df["soc"].mean()),
        "final_soc": float(battery_df["soc"].iloc[-1]),
        "total_buy_sp15": float(results_df.loc[results_df["product"] == "SP15", "buy_qty"].sum()),
        "total_sell_sp15": float(results_df.loc[results_df["product"] == "SP15", "sell_qty"].sum()),
        "total_hourly_profit": float(pnl_df["hourly_pnl"].sum()),
    }

    return {
        "label": label,
        "input_df": merged_df,
        "results_df": results_df,
        "battery_df": battery_df,
        "pnl_df": pnl_df,
        "summary": summary,
    }


def add_comparison_metrics(run_outputs):
    clean_soc = run_outputs["clean"]["battery_df"]["soc"].to_numpy()
    clean_profit = run_outputs["clean"]["summary"]["objective_profit"]

    for label, output in run_outputs.items():
        soc = output["battery_df"]["soc"].to_numpy()
        output["summary"]["soc_deviation_from_clean"] = float(np.mean(np.abs(soc - clean_soc)))
        output["summary"]["profit_change_from_clean"] = float(output["summary"]["objective_profit"] - clean_profit)

    return run_outputs


def percent_recovery(attack_val, method_val, clean_val):
    denom = attack_val - clean_val
    if abs(denom) < 1e-12:
        return 0.0
    return ((attack_val - method_val) / denom) * 100.0


def make_summary_table(run_outputs):
    rows = [run_outputs[k]["summary"] for k in run_outputs]
    summary_df = pd.DataFrame(rows)

    clean = summary_df.loc[summary_df["case"] == "clean"].iloc[0]
    attack = summary_df.loc[summary_df["case"] == "attack"].iloc[0]

    recovery_rows = []
    for case in ["zscore", "median", "iqr"]:
        row = summary_df.loc[summary_df["case"] == case].iloc[0]
        recovery_rows.append({
            "case": case,
            "profit_recovery_pct": percent_recovery(
                attack["objective_profit"], row["objective_profit"], clean["objective_profit"]
            ),
            "volatility_reduction_recovery_pct": percent_recovery(
                attack["price_volatility_mad"], row["price_volatility_mad"], clean["price_volatility_mad"]
            ),
            "soc_recovery_pct": percent_recovery(
                attack["soc_deviation_from_clean"], row["soc_deviation_from_clean"], 0.0
            ),
        })

    recovery_df = pd.DataFrame(recovery_rows)
    return summary_df, recovery_df


def save_case_plots(output):
    label = output["label"]
    df = output["input_df"].copy()
    battery_df = output["battery_df"].copy()
    pnl_df = output["pnl_df"].copy()

    df["datetime"] = pd.to_datetime(df["datetime"])
    battery_df["datetime"] = pd.to_datetime(battery_df["datetime"])
    pnl_df["datetime"] = pd.to_datetime(pnl_df["datetime"])

    plt.figure(figsize=(14, 5))
    plt.plot(df["datetime"], df["SP15"], linewidth=2)
    plt.title(f"SP15 Price - {label}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label}_price.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(battery_df["datetime"], battery_df["soc"], linewidth=2)
    plt.fill_between(battery_df["datetime"], 0, battery_df["soc"], alpha=0.25)
    plt.axhline(y=mcp, linestyle="--")
    plt.title(f"Battery SOC - {label}")
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label}_soc.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.bar(
        pnl_df["datetime"],
        pnl_df["hourly_pnl"],
        alpha=0.7,
    )
    plt.title(f"Hourly Profit - {label}")
    plt.xlabel("Time")
    plt.ylabel("Hourly Profit")
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label}_hourly_profit.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_comparison_plots(run_outputs):
    clean_df = run_outputs["clean"]["input_df"].copy()
    attack_df = run_outputs["attack"]["input_df"].copy()
    iqr_df = run_outputs["iqr"]["input_df"].copy()
    median_df = run_outputs["median"]["input_df"].copy()
    zscore_df = run_outputs["zscore"]["input_df"].copy()

    for d in [clean_df, attack_df, iqr_df, median_df, zscore_df]:
        d["datetime"] = pd.to_datetime(d["datetime"])

    plt.figure(figsize=(15, 6))
    plt.plot(clean_df["datetime"], clean_df["SP15"], label="clean", linewidth=2)
    plt.plot(attack_df["datetime"], attack_df["SP15"], label="attack", linewidth=2)
    plt.plot(zscore_df["datetime"], zscore_df["SP15"], label="zscore", linewidth=2)
    plt.plot(median_df["datetime"], median_df["SP15"], label="median", linewidth=2)
    plt.plot(iqr_df["datetime"], iqr_df["SP15"], label="iqr", linewidth=2)
    plt.title("SP15 Price Comparison")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_prices.png"), dpi=300, bbox_inches="tight")
    plt.close()

    labels = ["clean", "attack", "zscore", "median", "iqr"]
    profits = [run_outputs[k]["summary"]["objective_profit"] for k in labels]
    vol = [run_outputs[k]["summary"]["price_volatility_mad"] for k in labels]
    soc_dev = [run_outputs[k]["summary"]["soc_deviation_from_clean"] for k in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, profits)
    plt.title("Objective Profit Comparison")
    plt.ylabel("Profit")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_profit.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, vol)
    plt.title("Price Volatility (MAD) Comparison")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_volatility.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, soc_dev)
    plt.title("SOC Deviation from Clean")
    plt.ylabel("SOC Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_soc_deviation.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_csv_outputs(run_outputs, summary_df, recovery_df):
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_table.csv"), index=False)
    recovery_df.to_csv(os.path.join(OUTPUT_DIR, "recovery_table.csv"), index=False)

    for label, output in run_outputs.items():
        output["input_df"].to_csv(os.path.join(OUTPUT_DIR, f"{label}_input_prices.csv"), index=False)
        output["results_df"].to_csv(os.path.join(OUTPUT_DIR, f"{label}_dispatch.csv"), index=False)
        output["battery_df"].to_csv(os.path.join(OUTPUT_DIR, f"{label}_battery.csv"), index=False)
        output["pnl_df"].to_csv(os.path.join(OUTPUT_DIR, f"{label}_pnl.csv"), index=False)


def main():
    run_outputs = {
        "clean": build_and_solve_model(merged_df_clean, "clean"),
        "attack": build_and_solve_model(merged_df_spike, "attack"),
        "zscore": build_and_solve_model(merged_df_zscore, "zscore"),
        "median": build_and_solve_model(merged_df_median, "median"),
        "iqr": build_and_solve_model(merged_df_iqr, "iqr"),
    }

    run_outputs = add_comparison_metrics(run_outputs)
    summary_df, recovery_df = make_summary_table(run_outputs)

    print("\nSummary Table")
    print(summary_df.to_string(index=False))

    print("\nRecovery Table")
    print(recovery_df.to_string(index=False))

    save_csv_outputs(run_outputs, summary_df, recovery_df)

    for label in run_outputs:
        save_case_plots(run_outputs[label])

    save_comparison_plots(run_outputs)

    print(f"\nAll results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()