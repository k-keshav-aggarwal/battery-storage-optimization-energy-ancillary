"""Unsupervised anomaly detection: VAE + Isolation Forest + LOF ensemble.

Two leakage fixes relative to the notebook lineage:

* every model is fit on the training split only (this was already true), and
* every score is normalised using **training-split quantiles only**. The
  original normalised by ``score.max() - score.min()`` over the full series,
  so the evaluation window set the scale its own scores were measured against.

Causality
---------
Isolation Forest and LOF score each hour from a feature vector built only from
that hour and earlier, so they are strictly causal.

The VAE reconstructs a 24-hour block, so hour *t*'s error depends on other
hours in the same block. Blocks are aligned to the midnight-to-midnight
day-ahead market day, which the dispatcher also solves as a unit -- the entire
block is announced simultaneously, so this is information genuinely available
at decision time. It is *block*-causal, not strictly causal, and is labelled
as such rather than being quietly mixed in with the trailing features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .config import (
    ENSEMBLE_WEIGHTS,
    VAE_BATCH,
    VAE_EPOCHS,
    VAE_LATENT_DIM,
    VAE_LR,
    VAE_SEQ_LEN,
)


class VAE(nn.Module):
    """Small dense VAE over a 24-hour price block."""

    def __init__(self, input_dim: int = VAE_SEQ_LEN, latent_dim: int = VAE_LATENT_DIM, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, input_dim))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.dec(z), mu, logvar


def _vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    mse = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld / x.shape[1]


def _blocks(x: np.ndarray, seq_len: int = VAE_SEQ_LEN) -> np.ndarray:
    """Reshape a 1-D series into midnight-aligned blocks, zero-padding the tail."""
    rem = len(x) % seq_len
    if rem:
        x = np.concatenate([x, np.zeros(seq_len - rem)])
    return x.reshape(-1, seq_len)


@dataclass
class TrainStats:
    """Normalisation constants, fitted on the training split only."""

    lo: float
    hi: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        span = self.hi - self.lo
        return np.clip((x - self.lo) / (span if span > 1e-12 else 1.0), 0.0, 1.0)

    @classmethod
    def fit(cls, x_train: np.ndarray) -> "TrainStats":
        return cls(lo=float(np.quantile(x_train, 0.01)), hi=float(np.quantile(x_train, 0.99)))


@dataclass
class DetectorBundle:
    vae_score: np.ndarray
    iso_score: np.ndarray
    lof_score: np.ndarray
    ensemble: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"vae": self.vae_score, "iso": self.iso_score, "lof": self.lof_score}


def _fit_vae(price_scaled: np.ndarray, train_mask: np.ndarray, seed: int,
             epochs: int = VAE_EPOCHS) -> np.ndarray:
    """Train on training-split blocks, return a per-hour reconstruction error."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_blocks = _blocks(price_scaled)
    # A block belongs to training only if every hour in it does.
    block_train = _blocks(train_mask.astype(float)) .min(axis=1) > 0.5
    x_train = torch.tensor(all_blocks[block_train], dtype=torch.float32)

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=VAE_LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train), batch_size=VAE_BATCH, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            loss = _vae_loss(recon, batch, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        recon, _, _ = model(torch.tensor(all_blocks, dtype=torch.float32))
        err = (torch.tensor(all_blocks, dtype=torch.float32) - recon).pow(2).numpy()
    return err.flatten()[:len(price_scaled)]


def run_detectors(datetimes, price_observed: np.ndarray, train_mask: np.ndarray,
                  seed: int = 0, epochs: int = VAE_EPOCHS) -> DetectorBundle:
    """Fit all three detectors on the training split and score the full series."""
    from .features import build_causal_features

    price = np.asarray(price_observed, dtype=float)

    # Causal, scale-free inputs for the two classical detectors. Fitting them
    # on raw price alone (as the original did) leaves them unable to separate a
    # high-but-normal evening peak from an injected one.
    feats = build_causal_features(datetimes, price)
    x = feats[["f_robust_z", "f_ratio_long", "f_ramp"]].values
    scaler = StandardScaler().fit(x[train_mask])
    xs = scaler.transform(x)

    # --- VAE, on train-split min/max scaled prices --------------------------
    p_tr = price[train_mask]
    p_lo, p_hi = float(np.quantile(p_tr, 0.001)), float(np.quantile(p_tr, 0.999))
    p_scaled = np.clip((price - p_lo) / max(p_hi - p_lo, 1e-9), -1.0, 5.0)
    vae_raw = _fit_vae(p_scaled, train_mask, seed=seed, epochs=epochs)

    # --- Isolation Forest ---------------------------------------------------
    iso = IsolationForest(contamination=0.05, random_state=seed, n_estimators=200)
    iso.fit(xs[train_mask])
    iso_raw = -iso.score_samples(xs)

    # --- Local Outlier Factor ----------------------------------------------
    lof = LocalOutlierFactor(novelty=True, n_neighbors=20, contamination=0.05)
    lof.fit(xs[train_mask])
    lof_raw = -lof.score_samples(xs)

    # --- normalise using TRAIN quantiles only -------------------------------
    vae_s = TrainStats.fit(vae_raw[train_mask]).apply(vae_raw)
    iso_s = TrainStats.fit(iso_raw[train_mask]).apply(iso_raw)
    lof_s = TrainStats.fit(lof_raw[train_mask]).apply(lof_raw)

    w_v, w_i, w_l = ENSEMBLE_WEIGHTS
    ensemble = np.clip(w_v * vae_s + w_i * iso_s + w_l * lof_s, 0.0, 1.0)
    return DetectorBundle(vae_score=vae_s, iso_score=iso_s, lof_score=lof_s, ensemble=ensemble)
