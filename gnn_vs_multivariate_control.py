"""GNN vs. plain multivariate LSTM-AE -- the control this project's own
standard requires before "GNN helps" can be claimed.

Self-contained: only needs the two CAISO cache pickle files. Trains BOTH
models on the IDENTICAL data/splits/seeds in one run, so the comparison is
apples-to-apples without needing to cross-reference a separate script.

What this isolates: the GNN result (0.936 +/- 0.097 mean AUROC, verified
earlier) could come from (a) the graph attention mechanism genuinely
learning useful cross-node structure, or (b) simply having 3 nodes of
input instead of 1, regardless of architecture. The "plain multivariate"
model here gets the EXACT SAME 3-node input, same window length, same
training budget -- just fed directly into an LSTM with no attention layer
at all. If it also jumps to ~0.9+, the graph isn't doing the work. If it
stays much closer to the single-node LSTM-AE's 0.738, the graph mechanism
itself is what matters.

Usage:
    python gnn_vs_multivariate_control.py
    python gnn_vs_multivariate_control.py --data-dir /path/to/cache/files
"""
import argparse
import fnmatch
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

torch.set_num_threads(1)
SEEDS = [0, 1, 2, 3, 4]
TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-07-01")
LMP_PATTERN = "cache_lmp_20230101_*.pkl"
AS_PATTERN = "cache_as_20230101_*.pkl"
NODE_MAP = {"TH_NP15_GEN-APND": "NP15", "TH_SP15_GEN-APND": "SP15_node", "TH_ZP26_GEN-APND": "ZP26"}


# ---------------- Data loading (inlined, no repo dependency) ----------------

def find_data_dir(explicit=None):
    roots = [explicit] if explicit else []
    roots += [os.environ.get("CAISO_DATA_DIR", ""), "./data", ".", os.path.expanduser("~/Downloads")]
    for root in roots:
        if root and os.path.isdir(root):
            files = os.listdir(root)
            if any(fnmatch.fnmatch(f, LMP_PATTERN) for f in files) and any(fnmatch.fnmatch(f, AS_PATTERN) for f in files):
                return root
    raise FileNotFoundError(f"Could not find both cache files in {roots}. Place them in ./data or pass --data-dir.")


def _latest(files, pattern):
    matches = sorted(f for f in files if fnmatch.fnmatch(f, pattern))
    return matches[-1] if matches else None


def load_raw(data_dir):
    files = os.listdir(data_dir)
    lmp = pd.read_pickle(os.path.join(data_dir, _latest(files, LMP_PATTERN)))
    as_p = pd.read_pickle(os.path.join(data_dir, _latest(files, AS_PATTERN)))
    return lmp, as_p


def build_aggregated(data_dir):
    lmp, as_p = load_raw(data_dir)
    df = pd.merge(lmp, as_p, on="datetime", how="inner")
    return (df.groupby("datetime").agg({"SP15": "mean"}).reset_index()
            .sort_values("datetime").reset_index(drop=True))


def build_node_prices(data_dir):
    lmp, _ = load_raw(data_dir)
    wide = lmp.pivot_table(index="datetime", columns="node", values="SP15", aggfunc="mean")
    wide = wide.rename(columns=NODE_MAP).reset_index()
    missing = set(NODE_MAP.values()) - set(wide.columns)
    if missing:
        raise ValueError(f"Expected node columns {sorted(NODE_MAP.values())}, missing {missing}.")
    return wide.sort_values("datetime").reset_index(drop=True)


def reference_anomaly_flag(df, price_col="SP15", train_end=None):
    df = df.copy()
    price = df[price_col]
    roll_med = price.rolling(24 * 30, min_periods=48).median().bfill()
    roll_mad = (price - roll_med).abs().rolling(24 * 30, min_periods=48).median().bfill().replace(0, 1e-6)
    robust_z = (price - roll_med) / (1.4826 * roll_mad)
    train_prices = price[df["datetime"] < train_end] if train_end is not None else price
    p_low, p_high = train_prices.quantile(0.005), train_prices.quantile(0.995)
    df["ref_anomaly"] = ((robust_z.abs() > 4) | (price < p_low) | (price > p_high)).astype(int)
    return df


def split(df):
    train = df[df["datetime"] < TRAIN_END].reset_index(drop=True)
    val = df[(df["datetime"] >= TRAIN_END) & (df["datetime"] < VAL_END)].reset_index(drop=True)
    test = df[df["datetime"] >= VAL_END].reset_index(drop=True)
    return train, val, test


def node_windows(scaled_multi, seq_len=24):
    n, nodes = scaled_multi.shape
    rem = n % seq_len
    if rem:
        scaled_multi = np.vstack([scaled_multi, np.zeros((seq_len - rem, nodes))])
    return scaled_multi.reshape(-1, seq_len, nodes)


# ---------------- Model 1: GNN (graph attention + LSTM) ----------------

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, x):
        h = self.W(x)
        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        e = torch.nn.functional.leaky_relu(self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        alpha = torch.softmax(e, dim=-1)
        return torch.einsum("bij,bjd->bid", alpha, h), alpha


class GNNAutoencoder(nn.Module):
    def __init__(self, n_nodes=3, gat_hidden=8, lstm_hidden=32, latent=8, seq_len=24):
        super().__init__()
        self.gat1 = GraphAttentionLayer(1, gat_hidden)
        self.gat2 = GraphAttentionLayer(gat_hidden, gat_hidden)
        self.enc_lstm = nn.LSTM(n_nodes * gat_hidden, lstm_hidden, batch_first=True)
        self.enc_fc = nn.Linear(lstm_hidden, latent)
        self.dec_fc = nn.Linear(latent, lstm_hidden)
        self.dec_lstm = nn.LSTM(lstm_hidden, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, n_nodes)

    def forward(self, x):
        b, t, n = x.shape
        h, _ = self.gat1(x.reshape(b * t, n, 1))
        h = torch.relu(h)
        h, _ = self.gat2(h)
        h = torch.relu(h).reshape(b, t, n * h.shape[-1])
        _, (hn, _) = self.enc_lstm(h)
        z = self.enc_fc(hn[-1])
        dec_h0 = self.dec_fc(z).unsqueeze(1).repeat(1, t, 1)
        dec_out, _ = self.dec_lstm(dec_h0)
        return self.out(dec_out)


# ---------------- Model 2: plain multivariate LSTM-AE (the control -- NO graph/attention at all) ----------------

class MultivariateLSTMAutoencoder(nn.Module):
    """Identical in spirit to this project's existing single-node LSTM-AE,
    just with input/output width 3 instead of 1. No attention, no explicit
    node-pair modeling -- the LSTM itself is free to learn cross-node
    correlation implicitly if it can, exactly the way a naive "just
    concatenate the extra channels" approach would. This is the control:
    same data, same budget, no graph mechanism."""

    def __init__(self, n_nodes=3, hidden=32, latent=8, seq_len=24):
        super().__init__()
        self.enc = nn.LSTM(n_nodes, hidden, batch_first=True)
        self.enc_fc = nn.Linear(hidden, latent)
        self.dec_fc = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_nodes)
        self.seq_len = seq_len

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.enc_fc(h[-1])
        h0 = self.dec_fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.dec(h0)
        return self.out(dec_out)


# ---------------- Shared train/score/eval ----------------

def train_autoencoder(model_cls, train_windows, val_windows, epochs=40, batch=32, lr=1e-3):
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_tr = torch.tensor(train_windows, dtype=torch.float32)
    x_val = torch.tensor(val_windows, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x_tr.shape[0])
        for i in range(0, x_tr.shape[0], batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x_tr[idx]), x_tr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        val_loss = nn.functional.mse_loss(model(x_val), x_val).item()
    return model, val_loss


def score_windows(model, windows_arr):
    model.eval()
    x = torch.tensor(windows_arr, dtype=torch.float32)
    with torch.no_grad():
        err = (x - model(x)).pow(2).mean(dim=-1).numpy()
    return err


def best_budget(scores, labels, budgets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.10)):
    n = len(scores)
    best_f1, best_k = 0, budgets[0]
    for k in budgets:
        cut = int(np.ceil(n * k))
        thr = np.sort(scores)[-cut] if cut > 0 else scores.max() + 1
        f1 = f1_score(labels, (scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_k = f1, k
    return best_k


def eval_score(scored, col):
    _, val_s, test_s = split(scored)
    budget = best_budget(val_s[col].values, val_s["ref_anomaly"].values)
    ts, tl = test_s[col].values, test_s["ref_anomaly"].values
    cut = int(np.ceil(len(ts) * budget))
    thr = np.sort(ts)[-cut]
    pred = (ts >= thr).astype(int)
    return dict(
        budget=budget, AUROC=roc_auc_score(tl, ts), AP=average_precision_score(tl, ts),
        precision=precision_score(tl, pred, zero_division=0), recall=recall_score(tl, pred, zero_division=0),
        F1=f1_score(tl, pred, zero_division=0),
    )


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a - b
    wins = int((diffs > 0).sum())
    rng = np.random.default_rng(seed)
    boot = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(wins=wins, n=len(diffs), mean_diff=float(diffs.mean()), ci=(float(lo), float(hi)))


# ---------------- Main ----------------

def main(data_dir=None):
    data_dir = find_data_dir(data_dir)
    print(f"Using CAISO data directory: {data_dir}\n")

    agg = build_aggregated(data_dir)
    agg = reference_anomaly_flag(agg, train_end=TRAIN_END)
    nodes = build_node_prices(data_dir)
    agg_multi = agg.merge(nodes[["datetime", "NP15", "SP15_node", "ZP26"]], on="datetime", how="left")
    agg_multi[["NP15", "SP15_node", "ZP26"]] = agg_multi[["NP15", "SP15_node", "ZP26"]].ffill().bfill()
    print(f"Loaded {len(agg_multi)} hourly rows, 3 nodes.\n")

    train, val, test = split(agg_multi)
    node_cols = ["NP15", "SP15_node", "ZP26"]

    gnn_aurocs, mv_aurocs = [], []
    for seed in SEEDS:
        t0 = time.time()
        scaler = MinMaxScaler().fit(train[node_cols].values)
        w_train = node_windows(scaler.transform(train[node_cols].values))
        w_val = node_windows(scaler.transform(val[node_cols].values))
        w_all = node_windows(scaler.transform(agg_multi[node_cols].values))

        # GNN: own RNG stream, offset +50_000 (matches this project's per-model convention)
        torch.manual_seed(seed + 50_000); np.random.seed(seed + 50_000)
        gnn_model, _ = train_autoencoder(GNNAutoencoder, w_train, w_val, epochs=40)
        gnn_flat = score_windows(gnn_model, w_all).flatten()[:len(agg_multi)]
        gnn_rng = gnn_flat.max() - gnn_flat.min()
        gnn_score = (gnn_flat - gnn_flat.min()) / (gnn_rng if gnn_rng > 1e-8 else 1.0)

        # Plain multivariate LSTM: its OWN independent stream, offset +70_000 --
        # not +50_000 again, so it can't be silently affected by the GNN's RNG usage this round.
        torch.manual_seed(seed + 70_000); np.random.seed(seed + 70_000)
        mv_model, _ = train_autoencoder(MultivariateLSTMAutoencoder, w_train, w_val, epochs=40)
        mv_flat = score_windows(mv_model, w_all).flatten()[:len(agg_multi)]
        mv_rng = mv_flat.max() - mv_flat.min()
        mv_score = (mv_flat - mv_flat.min()) / (mv_rng if mv_rng > 1e-8 else 1.0)

        scored = agg_multi.copy()
        scored["gnn_score"], scored["mv_score"] = gnn_score, mv_score
        gnn_res = eval_score(scored, "gnn_score")
        mv_res = eval_score(scored, "mv_score")
        gnn_aurocs.append(gnn_res["AUROC"]); mv_aurocs.append(mv_res["AUROC"])
        print(f"seed {seed} done in {time.time()-t0:.1f}s: "
              f"GNN AUROC={gnn_res['AUROC']:.3f}  |  plain-multivariate LSTM AUROC={mv_res['AUROC']:.3f}")

    gnn_aurocs, mv_aurocs = np.array(gnn_aurocs), np.array(mv_aurocs)
    print(f"\nGNN (graph attention):        mean AUROC = {gnn_aurocs.mean():.3f} +/- {gnn_aurocs.std():.3f}")
    print(f"Plain multivariate LSTM-AE:    mean AUROC = {mv_aurocs.mean():.3f} +/- {mv_aurocs.std():.3f}")
    print(f"(reference) Single-node LSTM-AE (blended):  0.738 +/- 0.145  [already verified earlier]")

    boot = paired_bootstrap(gnn_aurocs, mv_aurocs)
    print(f"\nPaired bootstrap, GNN vs plain multivariate: wins={boot['wins']}/{boot['n']}  "
          f"mean_diff={boot['mean_diff']:+.3f}  95% CI=[{boot['ci'][0]:.3f}, {boot['ci'][1]:.3f}]")

    print("\n--- How to read this ---")
    if boot["ci"][0] > 0:
        print("CI excludes zero, favoring GNN: the graph attention mechanism appears to add real value")
        print("beyond just having 3 nodes of input -- worth continuing to Window 2 / disjoint-seed checks.")
    elif mv_aurocs.mean() > 0.85:
        print("Plain multivariate LSTM also scores very high: the gain looks like it's mostly from having")
        print("3 nodes of input at all, not specifically from the graph/attention mechanism.")
    else:
        print("Inconclusive at this sample size (n=5 seeds) -- see the CI width above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    main(data_dir=args.data_dir)
