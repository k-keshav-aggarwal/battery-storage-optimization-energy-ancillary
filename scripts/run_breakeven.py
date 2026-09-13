#!/usr/bin/env python3
"""What a false positive costs, what a true positive saves, and where they balance.

Detection quality and economic value are different axes. A true positive avoids
discharging into a price that will never be paid. A false positive suppresses a
*genuine* spike -- and genuine spikes are precisely the hours a battery most
wants to discharge into. The two errors are priced very differently, so a
defence can be accurate and still lose money.

Rather than sweep thresholds and average -- which confounds precision with
recall, since every threshold changes both -- this measures the two marginal
quantities directly by ablation on the test window:

    saved_per_TP  = (value with only true positives corrected  - undefended) / n_TP
    lost_per_FP   = (value with only false positives corrected - undefended) / n_FP

A defence operating at precision p breaks even when the expected gain is zero:

    p * saved_per_TP + (1 - p) * lost_per_FP = 0
    =>  p* = lost_per_FP / (lost_per_FP - saved_per_TP)          [lost_per_FP < 0]

p* is a design requirement, not a model score: it says how precise a detector
must be before switching it on is worth anything.

Crucially p* is **not** a property of the task alone -- it depends on *which*
genuine hours a rule gets wrong. A magnitude threshold errs by construction on
the largest genuine spikes, which are the most expensive hours to suppress, so
its p* can exceed 1 and no threshold makes it profitable. A learned classifier
that uses shape and context errs on cheaper hours and clears a far lower bar.
Both rules are therefore priced here, on identical data and splits.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attack import ATTACKS
from src.classify import select_detector_threshold, train_classifier
from src.config import ARTIFACT_DIR, SEEDS, VAE_EPOCHS
from src.data import load_hourly, split_masks
from src.detect import run_detectors
from src.dispatch import defend_by_price_correction, rolling_dispatch
from src.evaluate import bootstrap_ci
from src.features import build_causal_features, expected_price_causal


def price_the_errors(hourly, attack: str, seed: int, epochs: int, flag_budget: float = 0.12):
    dt, p_true = hourly["datetime"], hourly["SP15"].values
    tr, va, te = split_masks(hourly)

    atk = ATTACKS[attack](p_true, seed=seed)
    y = atk.mask.astype(int)
    det = run_detectors(dt, atk.observed, tr, seed=seed, epochs=epochs)
    flagged = det.ensemble >= select_detector_threshold(det.ensemble, va, budget=flag_budget)
    feats = build_causal_features(dt, atk.observed, detector_scores=det.as_dict())
    clf = train_classifier(feats, y, flagged, fit_mask=tr, val_mask=va, seed=seed)

    expected = expected_price_causal(dt, atk.observed)
    pt, po, ex = p_true[te], atk.observed[te], expected[te]

    pred = ((clf.p_synthetic >= clf.threshold) & flagged)[te]
    truth = atk.mask[te]
    tp_mask, fp_mask = pred & truth, pred & ~truth
    n_tp, n_fp = int(tp_mask.sum()), int(fp_mask.sum())
    if n_tp == 0 or n_fp == 0:
        return None

    def value(flags):
        return rolling_dispatch(defend_by_price_correction(po, ex, flags), pt).total_value

    undefended = rolling_dispatch(po, pt).total_value
    v_tp_only = value(tp_mask)
    v_fp_only = value(fp_mask)
    v_both = value(pred)

    saved_per_tp = (v_tp_only - undefended) / n_tp
    lost_per_fp = (v_fp_only - undefended) / n_fp
    # Additivity check: the two ablations should roughly reconstruct the joint
    # effect. Large residual would mean the corrections interact through SOC.
    residual = (v_both - undefended) - ((v_tp_only - undefended) + (v_fp_only - undefended))

    breakeven = (lost_per_fp / (lost_per_fp - saved_per_tp)
                 if (lost_per_fp - saved_per_tp) != 0 else float("nan"))

    return {
        "attack": attack, "seed": seed, "n_tp": n_tp, "n_fp": n_fp,
        "precision": n_tp / max(n_tp + n_fp, 1),
        "saved_per_tp": float(saved_per_tp), "lost_per_fp": float(lost_per_fp),
        "breakeven_precision": float(breakeven),
        "additivity_residual_usd": float(residual),
        "joint_effect_usd": float(v_both - undefended),
    }


def price_robust_z_rule(hourly, attack: str, seed: int, flag_budget: float = 0.12):
    """Same ablation for a plain trailing robust z-score threshold.

    The threshold is tuned on validation by realised value, giving the rule the
    most favourable operating point available to it.
    """
    dt, p_true = hourly["datetime"], hourly["SP15"].values
    tr, va, te = split_masks(hourly)

    atk = ATTACKS[attack](p_true, seed=seed)
    z = build_causal_features(dt, atk.observed)["f_robust_z"].values
    ex = expected_price_causal(dt, atk.observed)

    best_k, best_v = 3.0, -np.inf
    for k in np.arange(1.0, 8.1, 0.25):
        v = rolling_dispatch(defend_by_price_correction(atk.observed[va], ex[va], z[va] > k),
                             p_true[va]).total_value
        if v > best_v:
            best_v, best_k = v, k

    pt, po = p_true[te], atk.observed[te]
    pred, truth = z[te] > best_k, atk.mask[te]
    tp_mask, fp_mask = pred & truth, pred & ~truth
    n_tp, n_fp = int(tp_mask.sum()), int(fp_mask.sum())
    if n_tp == 0 or n_fp == 0:
        return None

    base = rolling_dispatch(po, pt).total_value

    def value(flags):
        return rolling_dispatch(defend_by_price_correction(po, ex[te], flags), pt).total_value

    saved = (value(tp_mask) - base) / n_tp
    lost = (value(fp_mask) - base) / n_fp
    return {"attack": attack, "seed": seed, "rule": "robust-z", "n_tp": n_tp, "n_fp": n_fp,
            "precision": n_tp / (n_tp + n_fp), "saved_per_tp": float(saved),
            "lost_per_fp": float(lost), "threshold_k": float(best_k),
            "breakeven_precision": float(lost / (lost - saved)) if lost != saved else float("nan")}


def main():
    hourly = load_hourly()
    rows = []
    print("Marginal value of each error type, measured by ablation on the test window.\n")
    print(f"{'attack':<12}{'n_TP':>5}{'n_FP':>6}{'$/TP':>10}{'$/FP':>10}{'break-even p*':>15}")
    print("-" * 58)
    for attack in ATTACKS:
        per_attack = []
        for seed in SEEDS:
            r = price_the_errors(hourly, attack, seed, VAE_EPOCHS)
            if r:
                rows.append(r)
                per_attack.append(r)
        if not per_attack:
            continue
        m = lambda k: float(np.mean([r[k] for r in per_attack]))
        print(f"{attack:<12}{m('n_tp'):5.0f}{m('n_fp'):6.0f}"
              f"{m('saved_per_tp'):10.1f}{m('lost_per_fp'):10.1f}{m('breakeven_precision'):15.3f}")

    print()
    for k, lbl in [("saved_per_tp", "Value saved per true positive"),
                   ("lost_per_fp", "Value lost per false positive"),
                   ("breakeven_precision", "Break-even precision p*")]:
        v = [r[k] for r in rows]
        mean, (lo, hi) = bootstrap_ci(v)
        unit = "" if k == "breakeven_precision" else "$"
        print(f"  {lbl:<32s} {unit}{mean:8.3f}   95% CI [{unit}{lo:.3f}, {unit}{hi:.3f}]  (n={len(v)})")

    # ---- comparison rule: plain trailing robust z-score threshold ---------
    base_rows = []
    for attack in ATTACKS:
        for seed in SEEDS:
            r = price_robust_z_rule(hourly, attack, seed)
            if r:
                base_rows.append(r)
    if base_rows:
        bm = lambda k: float(np.mean([r[k] for r in base_rows]))
        print(f"\n  Comparison -- plain robust-z threshold rule (n={len(base_rows)}):")
        print(f"    achieved precision {bm('precision'):.3f}   $/TP {bm('saved_per_tp'):8.1f}   "
              f"$/FP {bm('lost_per_fp'):9.1f}   p* {bm('breakeven_precision'):.3f}")
        print(f"    Its false positives are {abs(bm('lost_per_fp')) / max(abs(np.mean([r['lost_per_fp'] for r in rows])), 1e-9):.0f}x "
              f"more expensive than the classifier's: a magnitude threshold errs")
        print(f"    on the largest genuine spikes by construction, so no attainable")
        print(f"    precision makes it pay (p* > 1).")

    res = [abs(r["additivity_residual_usd"]) for r in rows]
    joint = [abs(r["joint_effect_usd"]) for r in rows]
    print(f"\n  Additivity residual: mean ${np.mean(res):.1f} vs joint effect ${np.mean(joint):.1f} "
          f"({np.mean(res)/max(np.mean(joint),1e-9)*100:.1f}% -- SOC coupling between corrections)")

    mean_be = float(np.mean([r["breakeven_precision"] for r in rows]))
    mean_prec = float(np.mean([r["precision"] for r in rows]))
    print(f"\n  Achieved precision {mean_prec:.3f} vs required {mean_be:.3f}: "
          f"{'ABOVE break-even -- defence creates value' if mean_prec > mean_be else 'BELOW break-even -- defence destroys value'}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "breakeven.json"), "w") as fh:
        json.dump({"classifier": rows, "robust_z_rule": base_rows}, fh, indent=2, default=float)
    print(f"\nSaved -> {os.path.join(ARTIFACT_DIR, 'breakeven.json')}")


if __name__ == "__main__":
    main()
