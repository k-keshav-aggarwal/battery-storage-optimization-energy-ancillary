"""
eval_gru_model.py
=================
Evaluates GRU Autoencoder against LSTM-AE, VAE, and other detectors on CAISO data.
Calculates:
  - Test Reconstruction RMSE ($/MWh and normalized)
  - Test Reconstruction MAE ($/MWh and normalized)
  - F1 Score
  - Precision
  - Recall
  - AUROC
  - Average Precision (AP)
Across all 5 seeds.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import MinMaxScaler

from scripts.gnn_vs_multivariate_control import (
    find_data_dir,
    build_aggregated,
    reference_anomaly_flag,
    split,
    TRAIN_END,
    VAL_END,
    SEEDS,
    best_budget,
)

torch.set_num_threads(4)  # Use 4 threads for speed

# ----------------- Models -----------------

class GRUAutoencoder(nn.Module):
    def __init__(self, in_dim=1, hidden=32, latent=8, seq_len=24):
        super().__init__()
        self.enc = nn.GRU(in_dim, hidden, batch_first=True)
        self.enc_fc = nn.Linear(hidden, latent)
        self.dec_fc = nn.Linear(latent, hidden)
        self.dec = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, in_dim)
        self.seq_len = seq_len

    def forward(self, x):
        _, h = self.enc(x)
        z = self.enc_fc(h[-1])
        h0 = self.dec_fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.dec(h0)
        return self.out(dec_out)

class LSTMAutoencoder(nn.Module):
    def __init__(self, in_dim=1, hidden=32, latent=8, seq_len=24):
        super().__init__()
        self.enc = nn.LSTM(in_dim, hidden, batch_first=True)
        self.enc_fc = nn.Linear(hidden, latent)
        self.dec_fc = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, in_dim)
        self.seq_len = seq_len

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.enc_fc(h[-1])
        h0 = self.dec_fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.dec(h0)
        return self.out(dec_out)

class VAE(nn.Module):
    def __init__(self, input_dim=24, latent_dim=6, hidden=48):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, input_dim))
    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.view(b, -1)
        h = self.enc(x_flat)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        out = self.dec(z)
        return out.view(b, t, d)

def train_model(model_cls, train_w, val_w, epochs=40, batch=32, lr=1e-3):
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_tr = torch.tensor(train_w, dtype=torch.float32)
    x_val = torch.tensor(val_w, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x_tr.shape[0])
        for i in range(0, x_tr.shape[0], batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            recon = model(x_tr[idx])
            loss = nn.functional.mse_loss(recon, x_tr[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model

def make_windows(arr, seq_len=24):
    n = len(arr)
    rem = n % seq_len
    if rem:
        pad = np.zeros((seq_len - rem, arr.shape[1]))
        arr = np.vstack([arr, pad])
    return arr.reshape(-1, seq_len, arr.shape[1])

def run_eval():
    data_dir = find_data_dir()
    agg = build_aggregated(data_dir)
    agg = reference_anomaly_flag(agg, train_end=TRAIN_END)
    train, val, test = split(agg)
    
    price_col = ["SP15"]
    scaler = MinMaxScaler()
    scaler.fit(train[price_col].values)
    
    scaled_tr = scaler.transform(train[price_col].values)
    scaled_val = scaler.transform(val[price_col].values)
    scaled_test = scaler.transform(test[price_col].values)
    scaled_all = scaler.transform(agg[price_col].values)
    
    w_tr = make_windows(scaled_tr)
    w_val = make_windows(scaled_val)
    w_all = make_windows(scaled_all)
    
    # Store results across seeds for GRU, LSTM, VAE
    models_to_test = {
        "GRU-AE": GRUAutoencoder,
        "LSTM-AE": LSTMAutoencoder,
        "VAE": VAE
    }
    
    results = {m: [] for m in models_to_test}
    
    # Actual test set prices for RMSE / MAE
    test_raw = test["SP15"].values
    test_len = len(test_raw)
    test_start_idx = len(train) + len(val)
    
    print(f"Evaluating models across seeds {SEEDS}...")
    for seed in SEEDS:
        for name, cls in models_to_test.items():
            torch.manual_seed(seed + 10_000)
            np.random.seed(seed + 10_000)
            t0 = time.time()
            model = train_model(cls, w_tr, w_val, epochs=40)
            
            # Predict
            model.eval()
            with torch.no_grad():
                recon_all_w = model(torch.tensor(w_all, dtype=torch.float32)).numpy()
            
            recon_all = recon_all_w.reshape(-1, 1)[:len(agg)]
            err_all = (scaled_all - recon_all) ** 2
            err_score = err_all.flatten()
            
            # Anomaly scoring
            scored = agg.copy()
            norm_score = (err_score - err_score.min()) / (err_score.max() - err_score.min() + 1e-8)
            scored["score"] = norm_score
            
            # Test window evaluation
            val_s = scored[(scored["datetime"] >= TRAIN_END) & (scored["datetime"] < VAL_END)]
            test_s = scored[scored["datetime"] >= VAL_END]
            
            budget = best_budget(val_s["score"].values, val_s["ref_anomaly"].values)
            ts = test_s["score"].values
            tl = test_s["ref_anomaly"].values
            cut = int(np.ceil(len(ts) * budget))
            thr = np.sort(ts)[-cut]
            pred = (ts >= thr).astype(int)
            
            # Reconstruction RMSE / MAE on physical $/MWh
            recon_test = recon_all[test_start_idx:test_start_idx + test_len]
            recon_price = scaler.inverse_transform(recon_test).flatten()
            
            rmse_phys = float(np.sqrt(mean_squared_error(test_raw, recon_price)))
            mae_phys = float(mean_absolute_error(test_raw, recon_price))
            rmse_norm = float(np.sqrt(mean_squared_error(scaled_test.flatten(), recon_test.flatten())))
            mae_norm = float(mean_absolute_error(scaled_test.flatten(), recon_test.flatten()))
            
            auroc = float(roc_auc_score(tl, ts))
            ap = float(average_precision_score(tl, ts))
            p = float(precision_score(tl, pred, zero_division=0))
            r = float(recall_score(tl, pred, zero_division=0))
            f1 = float(f1_score(tl, pred, zero_division=0))
            
            res_item = {
                "seed": seed,
                "rmse_phys": rmse_phys,
                "mae_phys": mae_phys,
                "rmse_norm": rmse_norm,
                "mae_norm": mae_norm,
                "auroc": auroc,
                "ap": ap,
                "precision": p,
                "recall": r,
                "f1": f1,
                "budget": budget
            }
            results[name].append(res_item)
            print(f"[{name}] Seed {seed} ({time.time()-t0:.1f}s): AUC={auroc:.3f}, F1={f1:.3f}, RMSE=${rmse_phys:.2f}, MAE=${mae_phys:.2f}")

    print("\n=== SUMMARY METRICS (Mean +/- Std) ===")
    summary = {}
    for name, r_list in results.items():
        summary[name] = {
            "RMSE_phys": (float(np.mean([x["rmse_phys"] for x in r_list])), float(np.std([x["rmse_phys"] for x in r_list]))),
            "MAE_phys": (float(np.mean([x["mae_phys"] for x in r_list])), float(np.std([x["mae_phys"] for x in r_list]))),
            "RMSE_norm": (float(np.mean([x["rmse_norm"] for x in r_list])), float(np.std([x["rmse_norm"] for x in r_list]))),
            "MAE_norm": (float(np.mean([x["mae_norm"] for x in r_list])), float(np.std([x["mae_norm"] for x in r_list]))),
            "AUROC": (float(np.mean([x["auroc"] for x in r_list])), float(np.std([x["auroc"] for x in r_list]))),
            "AP": (float(np.mean([x["ap"] for x in r_list])), float(np.std([x["ap"] for x in r_list]))),
            "F1": (float(np.mean([x["f1"] for x in r_list])), float(np.std([x["f1"] for x in r_list]))),
            "Precision": (float(np.mean([x["precision"] for x in r_list])), float(np.std([x["precision"] for x in r_list]))),
            "Recall": (float(np.mean([x["recall"] for x in r_list])), float(np.std([x["recall"] for x in r_list]))),
        }
        s = summary[name]
        print(f"\nModel: {name}")
        print(f"  AUROC:      {s['AUROC'][0]:.3f} +/- {s['AUROC'][1]:.3f}")
        print(f"  F1 Score:   {s['F1'][0]:.3f} +/- {s['F1'][1]:.3f}")
        print(f"  Precision:  {s['Precision'][0]:.3f} +/- {s['Precision'][1]:.3f}")
        print(f"  Recall:     {s['Recall'][0]:.3f} +/- {s['Recall'][1]:.3f}")
        print(f"  RMSE:      ${s['RMSE_phys'][0]:.2f} +/- ${s['RMSE_phys'][1]:.2f} (norm: {s['RMSE_norm'][0]:.4f})")
        print(f"  MAE:       ${s['MAE_phys'][0]:.2f} +/- ${s['MAE_phys'][1]:.2f} (norm: {s['MAE_norm'][0]:.4f})")
        
    with open("artifacts/gru_eval_summary.json", "w") as f:
        json.dump({"runs": results, "summary": summary}, f, indent=2)
    print("\nSaved artifacts/gru_eval_summary.json successfully!")

if __name__ == "__main__":
    run_eval()
