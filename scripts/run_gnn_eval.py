import sys, os, time, json
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch

from src.data import build_aggregated, build_node_prices_and_spreads, split, TRAIN_END
from src.features import add_time_features, reference_anomaly_flag
import src.evaluation as ev
from src.config import SEEDS
from sklearn.preprocessing import MinMaxScaler
from gnn_detector import train_gnn_ae, gnn_ae_score, node_windows

torch.set_num_threads(1)
data_dir = "."

# eval_split/eval_ensemble read SCORE_COLS from src.evaluation's namespace at
# call time -- patch it BEFORE the seed loop, not after, so every eval_split
# call inside the loop (not just eval_ensemble at the end) sees "gnn_score".
import src.evaluation as ev
ev.SCORE_COLS = ["gnn_score"]

print("Loading real CAISO data...")
agg = build_aggregated(data_dir)
agg = add_time_features(agg)
agg = reference_anomaly_flag(agg, train_end=TRAIN_END)

node_spreads = build_node_prices_and_spreads(data_dir)
agg_multi = agg.merge(node_spreads[["datetime", "NP15", "SP15_node", "ZP26"]], on="datetime", how="left")
agg_multi[["NP15", "SP15_node", "ZP26"]] = agg_multi[["NP15", "SP15_node", "ZP26"]].ffill().bfill()
print(f"Merged multi-node data: {len(agg_multi)} rows, node columns present: "
      f"{[c for c in ['NP15','SP15_node','ZP26'] if c in agg_multi.columns]}")

train, val, test = split(agg_multi)
node_cols = ["NP15", "SP15_node", "ZP26"]

all_runs, all_scored = [], []
for seed in SEEDS:
    t0 = time.time()
    torch.manual_seed(seed + 50_000)  # own RNG stream offset, following this project's per-model convention
    np.random.seed(seed + 50_000)

    scaler = MinMaxScaler()
    scaler.fit(train[node_cols].values)
    scaled_train = scaler.transform(train[node_cols].values)
    scaled_val = scaler.transform(val[node_cols].values)
    scaled_all = scaler.transform(agg_multi[node_cols].values)

    w_train = node_windows(scaled_train)
    w_val = node_windows(scaled_val)
    w_all = node_windows(scaled_all)

    model, val_loss = train_gnn_ae(w_train, w_val, epochs=40)
    err = gnn_ae_score(model, w_all)
    flat = err.flatten()[:len(agg_multi)]
    rng = flat.max() - flat.min()
    gnn_score = (flat - flat.min()) / (rng if rng > 1e-8 else 1.0)

    scored = agg_multi.copy()
    scored["gnn_score"] = gnn_score
    res = ev.eval_split(scored)  # reuses the SAME eval code as the other 4 detectors -- direct comparability
    all_runs.append(res)
    all_scored.append(scored)
    print(f"seed {seed} done in {time.time()-t0:.1f}s: GNN AUROC={res['gnn_score']['AUROC']:.3f}  "
          f"(val_loss={val_loss:.4f})")

with open("gnn_all_runs.json", "w") as f:
    json.dump(all_runs, f, indent=2, default=float)

aurocs = [r["gnn_score"]["AUROC"] for r in all_runs]
print(f"\nGNN detector: mean AUROC = {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}")

# Ensemble
ens_report, ens_df = ev.eval_ensemble(agg_multi, all_scored)
print(f"GNN ensemble AUROC = {ens_report['gnn_score']['AUROC']:.3f}  "
      f"AP = {ens_report['gnn_score']['AP']:.3f}")

with open("gnn_ensemble.json", "w") as f:
    json.dump(ens_report, f, indent=2, default=float)

print("\n=== Comparison against the already-verified Table II (same test window, same reference flag) ===")
print(f"{'Detector':20s} {'Mean AUROC':>12s} {'Ens AUROC':>10s}")
print(f"{'iForest':20s} {'0.618±0.020':>12s} {'0.618':>10s}   (from repo's verified run)")
print(f"{'VAE':20s} {'0.534±0.164':>12s} {'0.581':>10s}   (from repo's verified run)")
print(f"{'Diffusion':20s} {'0.564±0.071':>12s} {'0.631':>10s}   (from repo's verified run)")
print(f"{'LSTM-AE':20s} {'0.738±0.145':>12s} {'0.857':>10s}   (from repo's verified run)")
print(f"{'GNN (new)':20s} {np.mean(aurocs):9.3f}±{np.std(aurocs):.3f} {ens_report['gnn_score']['AUROC']:10.3f}   (this run, real data)")
