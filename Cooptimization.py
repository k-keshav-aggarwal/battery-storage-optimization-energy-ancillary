import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyomo.environ as pyo
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from pull_prices import merged_df_clean, merged_df_spike
from params import nodes, mcp, mdp, e, fee


# =========================
# 🔹 TRANSFORMER
# =========================
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 32)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 4, batch_first=True), 2
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        return self.out(self.tr(self.fc(x)))


def train_transformer(df):
    series = df["SP15"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    model = Transformer()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(3):
        pred = model(x)
        loss = ((pred - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    df["expected_price"] = scaler.inverse_transform(
        pred.detach().numpy().reshape(-1, 1)
    )

    return df


# =========================
# 🔹 GRAPH (LEARNED)
# =========================
def build_graph(df):

    pivot = df.pivot(index="datetime", columns="node", values="SP15")
    pivot = pivot.ffill()

    A = pivot.corr().values
    np.fill_diagonal(A, 0)

    D = np.diag(A.sum(axis=1))
    L = D - A

    return L


# =========================
# 🔹 ANOMALY DETECTION
# =========================
def compute_anomaly(df):

    X = df[["SP15"]].values

    iso = IsolationForest(contamination=0.05)
    lof = LocalOutlierFactor()

    df["iso"] = (iso.fit_predict(X) == -1).astype(int)
    df["lof"] = (lof.fit_predict(X) == -1).astype(int)

    df["res"] = abs(df["SP15"] - df["expected_price"])

    scaler = MinMaxScaler()
    df[["res"]] = scaler.fit_transform(df[["res"]])

    df["anomaly"] = df["res"] + df["iso"] + df["lof"]
    df["anomaly"] = MinMaxScaler().fit_transform(df[["anomaly"]])

    return df


# =========================
# 🔹 OPTIMIZATION (WITH SOC OUTPUT)
# =========================
def optimize(df, L, anomaly_on=True):

    model = pyo.ConcreteModel()
    T = len(df)

    model.t = pyo.RangeSet(0, T - 1)
    model.n = pyo.Set(initialize=nodes)

    model.price = pyo.Param(model.t, initialize=lambda m, t: df.loc[t, "SP15"])
    model.anom = pyo.Param(model.t, initialize=lambda m, t: df.loc[t, "anomaly"])

    model.buy = pyo.Var(model.t, model.n, bounds=(0, mcp))
    model.sell = pyo.Var(model.t, model.n, bounds=(0, mdp))
    model.soc = pyo.Var(model.t, bounds=(0, mcp))

    # SOC dynamics
    def soc_rule(m, t):
        if t == 0:
            return m.soc[t] == mcp
        return m.soc[t] == m.soc[t - 1] + \
            sum(m.buy[t, n] for n in m.n) * e - \
            sum(m.sell[t, n] for n in m.n) / e

    model.soc_c = pyo.Constraint(model.t, rule=soc_rule)

    # Objective
    def obj(m):
        profit = 0
        for t in m.t:
            w = 1 + (m.anom[t] if anomaly_on else 0)
            for n in m.n:
                profit += m.sell[t, n] * m.price[t] * w \
                          - m.buy[t, n] * m.price[t] / w

        return profit

    model.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

    pyo.SolverFactory("highs").solve(model)

    # Extract SOC
    soc = [pyo.value(model.soc[t]) for t in model.t]

    return pyo.value(model.obj), soc


# =========================
# 🔹 PLOTTING
# =========================
def plot_prices(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["SP15"].values, label="Actual Price")
    plt.plot(df["expected_price"].values, label="Expected Price")
    plt.title("Price vs Expected")
    plt.legend()
    plt.grid()
    plt.show()


def plot_anomaly(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df["anomaly"].values, color="red")
    plt.title("Anomaly Score")
    plt.grid()
    plt.show()


def plot_soc(soc, title):
    plt.figure(figsize=(12, 4))
    plt.plot(soc)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.grid()
    plt.show()


# =========================
# 🔹 PIPELINE
# =========================
def run(df):

    df = train_transformer(df)
    df = compute_anomaly(df)
    L = build_graph(df)

    base_profit, base_soc = optimize(df, L, False)
    anom_profit, anom_soc = optimize(df, L, True)

    return df, base_profit, anom_profit, base_soc, anom_soc


def plot_price_comparison(df):

    plt.figure(figsize=(12,5))
    plt.plot(df["SP15"].values, label="Actual Price")
    plt.plot(df["expected_price"].values, label="Expected Price")

    plt.title("Price vs Expected Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

def plot_soc_comparison(base_soc, anom_soc):

    plt.figure(figsize=(12,5))

    plt.plot(base_soc, label="Baseline SOC")
    plt.plot(anom_soc, label="Anomaly-Aware SOC")

    plt.title("SOC Comparison")
    plt.xlabel("Time")
    plt.ylabel("State of Charge")
    plt.legend()
    plt.grid()
    plt.show()

def plot_volatility(df):

    # rolling volatility
    df["volatility"] = df["SP15"].rolling(24).std()

    plt.figure(figsize=(12,4))
    plt.plot(df["volatility"], color="purple")

    plt.title("Price Volatility (Rolling Std)")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.grid()
    plt.show()

def plot_profit(clean_base, clean_anom, attack_base, attack_anom):

    labels = ["Clean Base", "Clean Anom", "Attack Base", "Attack Anom"]
    values = [clean_base, clean_anom, attack_base, attack_anom]

    plt.figure(figsize=(8,5))
    plt.bar(labels, values)

    plt.title("Profit Comparison")
    plt.ylabel("Profit")
    plt.grid(axis="y")
    plt.show()


# =========================
# 🔹 MAIN
# =========================
if __name__ == "__main__":

    print("Running CLEAN scenario...")
    clean_df, c_base, c_anom, c_soc_base, c_soc_anom = run(merged_df_clean)

    print("Running ATTACK scenario...")
    attack_df, a_base, a_anom, a_soc_base, a_soc_anom = run(merged_df_spike)

    print("\n===== RESULTS =====")
    print("Clean:", c_base, c_anom)
    print("Attack:", a_base, a_anom)

    # 🔥 PLOTS
    plot_price_comparison(clean_df)
    plot_volatility(clean_df)
    plot_soc_comparison(c_soc_base, c_soc_anom)
    plot_profit(c_base, c_anom, a_base, a_anom)