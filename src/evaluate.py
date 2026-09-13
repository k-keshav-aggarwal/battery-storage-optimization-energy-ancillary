"""Metrics and uncertainty reporting."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI on the mean of a small sample."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), (float(lo), float(hi))


def summarise(name: str, values, unit: str = "", pct: bool = False) -> str:
    v = np.asarray(values, dtype=float)
    mean, (lo, hi) = bootstrap_ci(v)
    fmt = (lambda x: f"{x:+.2f}%") if pct else (lambda x: f"{unit}{x:,.4f}".rstrip("0").rstrip("."))
    return f"{name:<34s} {fmt(mean):>12s}  95% CI [{fmt(lo)}, {fmt(hi)}]  (n={len(v)}, sd={v.std():.4f})"


def recovery_pct(defended: float, undefended: float, clean: float) -> float:
    """Share of attack damage recovered, 0% = no better than undefended."""
    damage = clean - undefended
    return float((defended - undefended) / damage * 100.0) if abs(damage) > 1e-9 else 0.0


def classification_report_at(p_synthetic, y_synthetic, flagged, mask, threshold):
    idx = np.where(mask & flagged)[0]
    y, p = y_synthetic[idx], p_synthetic[idx]
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "n_flagged": int(pred.sum()),
    }
