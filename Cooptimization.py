import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from pull_prices import (
    merged_df_clean,
    merged_df_attack,
    merged_df_zscore,
    merged_df_median,
    merged_df_iqr,
)
from params import mcp, mdp, e, fee, nodes, products

output_dir = os.path.join(os.getcwd(), "optimization_results")
os.makedirs(output_dir, exist_ok=True)


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


def build_model(price_df):
    df = price_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    model = pyo.ConcreteModel()
    model.t = pyo.Set(initialize=range(1, len(df) + 1))
    model.p = pyo.Set(initialize=products)
    model.n = pyo.Set(initialize=nodes)

    model.price = pyo.Param(model.t, model.p, model.n, mutable=True)
    for t in model.t:
        for p in model.p:
            for n in model.n:
                model.price[t, p, n] = float(df.loc[t - 1, p])

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
        return model.buy[t, p, n] <= mcp * model.is_buy[t, n]

    def buy_sell_exclusivity_sell(model, t, p, n):
        return model.sell[t, p, n] <= mdp * (1 - model.is_buy[t, n])

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
    return model, df


def solve_case(opt_df, label):
    model, df = build_model(opt_df)

    solver_name, solver = get_available_solver()
    print(f"Running {label} with solver: {solver_name}")
    results = solver.solve(model, tee=False, options=get_solver_options(solver_name))

    if results.solver.status != pyo.SolverStatus.ok:
        raise RuntimeError(
            f"{label}: solve failed | status={results.solver.status} "
            f"| termination={results.solver.termination_condition}"
        )

    dispatch_rows = []
    for t in model.t:
        for p in model.p:
            for n in model.n:
                dispatch_rows.append({
                    "case": label,
                    "time_step": t,
                    "datetime": df.loc[t - 1, "datetime"],
                    "product": p,
                    "node": n,
                    "opt_price": float(model.price[t, p, n].value),
                    "buy_qty": float(model.buy[t, p, n].value),
                    "sell_qty": float(model.sell[t, p, n].value),
                })

    dispatch_df = pd.DataFrame(dispatch_rows)

    battery_rows = []
    for t in model.t:
        battery_rows.append({
            "case": label,
            "time_step": t,
            "datetime": df.loc[t - 1, "datetime"],
            "soc": float(model.soc[t].value),
            "net_flow": float(model.net_energy_flow[t].value),
        })

    battery_df = pd.DataFrame(battery_rows)

    return {
        "label": label,
        "opt_df": df,
        "dispatch_df": dispatch_df,
        "battery_df": battery_df,
        "objective_profit_on_opt_prices": float(pyo.value(model.objective)),
    }


def evaluate_dispatch_on_prices(dispatch_df, eval_df):
    eval_df = eval_df.copy().reset_index(drop=True)
    eval_df["datetime"] = pd.to_datetime(eval_df["datetime"])

    merged = dispatch_df.merge(
        eval_df[["datetime"] + products],
        on="datetime",
        how="left",
        suffixes=("", "_eval")
    )

    pnl_rows = []
    for _, row in merged.iterrows():
        product = row["product"]
        eval_price = float(row[product])

        sell_revenue = float(row["sell_qty"]) * (eval_price - fee)
        buy_cost = float(row["buy_qty"]) * (eval_price + fee)
        pnl_rows.append(sell_revenue - buy_cost)

    merged["eval_pnl"] = pnl_rows

    hourly = merged.groupby("datetime", as_index=False)["eval_pnl"].sum()
    hourly["cumulative_eval_pnl"] = hourly["eval_pnl"].cumsum()

    total_profit = float(hourly["eval_pnl"].sum())
    return merged, hourly, total_profit


def total_decision_deviation(clean_dispatch_df, other_dispatch_df):
    a = clean_dispatch_df.sort_values(["time_step", "product", "node"]).reset_index(drop=True)
    b = other_dispatch_df.sort_values(["time_step", "product", "node"]).reset_index(drop=True)

    buy_dev = np.abs(a["buy_qty"].to_numpy() - b["buy_qty"].to_numpy()).sum()
    sell_dev = np.abs(a["sell_qty"].to_numpy() - b["sell_qty"].to_numpy()).sum()
    return float(buy_dev + sell_dev)


def soc_deviation(clean_battery_df, other_battery_df):
    a = clean_battery_df.sort_values("time_step")["soc"].to_numpy()
    b = other_battery_df.sort_values("time_step")["soc"].to_numpy()
    return float(np.mean(np.abs(a - b)))


def price_volatility_mad(df, col="SP15"):
    s = df[col]
    return float(np.mean(np.abs(s - s.mean())))


def make_summary(run_outputs):
    clean_eval_df = merged_df_clean.copy()
    clean_eval_df["datetime"] = pd.to_datetime(clean_eval_df["datetime"])

    summary_rows = []

    clean_dispatch = run_outputs["clean"]["dispatch_df"]
    clean_battery = run_outputs["clean"]["battery_df"]

    for label, out in run_outputs.items():
        eval_detail, eval_hourly, true_profit = evaluate_dispatch_on_prices(out["dispatch_df"], clean_eval_df)

        out["eval_detail_df"] = eval_detail
        out["eval_hourly_df"] = eval_hourly
        out["true_profit_on_clean_prices"] = true_profit

        summary_rows.append({
            "case": label,
            "objective_profit_on_opt_prices": out["objective_profit_on_opt_prices"],
            "true_profit_on_clean_prices": true_profit,
            "price_volatility_mad": price_volatility_mad(out["opt_df"], "SP15"),
            "soc_deviation_from_clean": soc_deviation(clean_battery, out["battery_df"]),
            "decision_deviation_from_clean": total_decision_deviation(clean_dispatch, out["dispatch_df"]),
        })

    return pd.DataFrame(summary_rows)


def save_results(run_outputs, summary_df):
    summary_path = os.path.join(output_dir, "summary_true_evaluation.csv")
    summary_df.to_csv(summary_path, index=False)

    for label, out in run_outputs.items():
        out["dispatch_df"].to_csv(os.path.join(output_dir, f"{label}_dispatch.csv"), index=False)
        out["battery_df"].to_csv(os.path.join(output_dir, f"{label}_battery.csv"), index=False)
        out["eval_hourly_df"].to_csv(os.path.join(output_dir, f"{label}_true_profit_hourly.csv"), index=False)
        out["opt_df"].to_csv(os.path.join(output_dir, f"{label}_input_prices.csv"), index=False)

    labels = ["clean", "attack", "zscore", "median", "iqr"]

    plt.figure(figsize=(10, 5))
    vals = [summary_df.loc[summary_df["case"] == x, "true_profit_on_clean_prices"].iloc[0] for x in labels]
    plt.bar(labels, vals)
    plt.title("True Profit on Clean Prices")
    plt.ylabel("Profit")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "true_profit_on_clean_prices.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    vals = [summary_df.loc[summary_df["case"] == x, "decision_deviation_from_clean"].iloc[0] for x in labels]
    plt.bar(labels, vals)
    plt.title("Decision Deviation from Clean")
    plt.ylabel("Total |buy-sell difference|")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_deviation_from_clean.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    vals = [summary_df.loc[summary_df["case"] == x, "soc_deviation_from_clean"].iloc[0] for x in labels]
    plt.bar(labels, vals)
    plt.title("SOC Deviation from Clean")
    plt.ylabel("Mean Absolute SOC Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "soc_deviation_from_clean_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    vals = [summary_df.loc[summary_df["case"] == x, "price_volatility_mad"].iloc[0] for x in labels]
    plt.bar(labels, vals)
    plt.title("Input Price Volatility (MAD)")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "price_volatility_mad_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(15, 6))
    for label in labels:
        df = run_outputs[label]["opt_df"].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        plt.plot(df["datetime"], df["SP15"], label=label, linewidth=2)
    plt.title("SP15 Price Comparison")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_prices_fixed.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    run_outputs = {
        "clean": solve_case(merged_df_clean, "clean"),
        "attack": solve_case(merged_df_attack, "attack"),
        "zscore": solve_case(merged_df_zscore, "zscore"),
        "median": solve_case(merged_df_median, "median"),
        "iqr": solve_case(merged_df_iqr, "iqr"),
    }

    summary_df = make_summary(run_outputs)
    print("\nFixed Evaluation Summary")
    print(summary_df.to_string(index=False))

    save_results(run_outputs, summary_df)
    print(f"\nSaved results in: {output_dir}")


if __name__ == "__main__":
    main()