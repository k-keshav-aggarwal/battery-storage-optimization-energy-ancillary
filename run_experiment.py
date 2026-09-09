#!/usr/bin/env python3
"""End-to-end experiment: detection -> classification -> day-ahead dispatch.

Protocol (identical for every seed and attack family):
  * detectors fit on TRAIN, score normalised with TRAIN quantiles
  * detector threshold chosen on VALIDATION at a fixed flag budget
  * classifier fit on TRAIN+VAL flagged hours, operating point chosen on VAL
  * every number reported comes from the TEST window, which nothing was fit on

Usage:
    python3 run_experiment.py                       # naive attack, all seeds
    python3 run_experiment.py --attack adaptive
    python3 run_experiment.py --seeds 0 1 --epochs 15
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.attack import ATTACKS
from src.classify import (auroc_on, conservative_threshold, select_detector_threshold,
                          train_classifier)
from src.config import ARTIFACT_DIR, SEEDS, VAE_EPOCHS
from src.data import load_hourly, split_masks
from src.detect import run_detectors
from src.dispatch import defend_by_clipping, defend_by_price_correction, rolling_dispatch
from src.evaluate import classification_report_at, recovery_pct, summarise
from src.features import build_causal_features, expected_price_causal


def run_seed(hourly, attack_name: str, seed: int, epochs: int, flag_budget: float):
    dt = hourly["datetime"]
    p_true = hourly["SP15"].values
    tr, va, te = split_masks(hourly)

    atk = ATTACKS[attack_name](p_true, seed=seed)
    p_obs, y = atk.observed, atk.mask.astype(int)

    det = run_detectors(dt, p_obs, tr, seed=seed, epochs=epochs)
    thr_det = select_detector_threshold(det.ensemble, va, budget=flag_budget)
    flagged = det.ensemble >= thr_det

    feats = build_causal_features(dt, p_obs, detector_scores=det.as_dict())
    clf = train_classifier(feats, y, flagged, fit_mask=tr, val_mask=va, seed=seed)
    thr_cons = conservative_threshold(clf.p_synthetic, y, flagged, va, target_precision=0.99)

    expected = expected_price_causal(dt, p_obs)

    # ---- dispatch on the test window ------------------------------------
    s = te
    pt, po, ex = p_true[s], p_obs[s], expected[s]

    def value(price_decision, discharge_limit=None):
        return rolling_dispatch(price_decision, pt, discharge_limit=discharge_limit).total_value

    v_clean = value(pt)
    v_naive = value(po)
    v_pen = value(po, discharge_limit=defend_by_clipping(det.ensemble[s], lam=0.3))
    v_f1 = value(defend_by_price_correction(po, ex, (clf.p_synthetic >= clf.threshold)[s]))
    v_cons = value(defend_by_price_correction(po, ex, (clf.p_synthetic >= thr_cons)[s]))
    v_oracle = value(defend_by_price_correction(po, ex, atk.mask[s]))

    rep_f1 = classification_report_at(clf.p_synthetic, y, flagged, te, clf.threshold)
    rep_cons = classification_report_at(clf.p_synthetic, y, flagged, te, thr_cons)

    return {
        "seed": seed,
        "detect_recall": float((flagged & (y == 1))[te].sum() / max((y[te] == 1).sum(), 1)),
        "detect_flag_rate": float(flagged[te].mean()),
        "clf_auroc": auroc_on(clf.p_synthetic, y, flagged, te),
        "clf_f1": rep_f1["f1"], "clf_precision": rep_f1["precision"], "clf_recall": rep_f1["recall"],
        "cons_precision": rep_cons["precision"], "cons_recall": rep_cons["recall"],
        "cons_tp": rep_cons["tp"], "cons_fp": rep_cons["fp"],
        "value_clean": v_clean, "value_naive": v_naive, "value_penalised": v_pen,
        "value_classifier": v_f1, "value_conservative": v_cons, "value_oracle": v_oracle,
        "damage_pct": (v_naive - v_clean) / abs(v_clean) * 100,
        "rec_penalised": recovery_pct(v_pen, v_naive, v_clean),
        "rec_classifier": recovery_pct(v_f1, v_naive, v_clean),
        "rec_conservative": recovery_pct(v_cons, v_naive, v_clean),
        "rec_oracle": recovery_pct(v_oracle, v_naive, v_clean),
        "importance": clf.feature_importance,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="naive", choices=list(ATTACKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    ap.add_argument("--flag-budget", type=float, default=0.12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hourly = load_hourly()
    print(f"Attack family : {args.attack}")
    print(f"Seeds         : {args.seeds}")
    print(f"Test window   : {hourly.datetime[split_masks(hourly)[2]].min()} -> {hourly.datetime.max()}\n")

    runs = []
    for seed in args.seeds:
        r = run_seed(hourly, args.attack, seed, args.epochs, args.flag_budget)
        runs.append(r)
        print(f"  seed {seed}: detect_recall={r['detect_recall']:.3f}  clf_AUROC={r['clf_auroc']:.3f}  "
              f"damage={r['damage_pct']:+.2f}%  recovered(cons)={r['rec_conservative']:.1f}%  "
              f"recovered(F1)={r['rec_classifier']:.1f}%")

    def col(k):
        return [r[k] for r in runs]

    print(f"\n{'='*88}\nRESULTS -- {args.attack} attack, test window, {len(runs)} seeds\n{'='*88}")
    print("\n-- Detection & classification (test window only) --")
    for k, lbl in [("detect_recall", "Detector recall"),
                   ("detect_flag_rate", "Detector flag rate"),
                   ("clf_auroc", "Classifier AUROC"),
                   ("clf_precision", "Classifier precision @F1"),
                   ("clf_recall", "Classifier recall @F1"),
                   ("cons_precision", "Classifier precision @conservative"),
                   ("cons_recall", "Classifier recall @conservative")]:
        print("  " + summarise(lbl, col(k)))

    print("\n-- Economics (realised at true settlement prices) --")
    for k, lbl in [("value_clean", "No attack"), ("value_naive", "Attacked, undefended"),
                   ("value_penalised", "Blanket penalty (lam=0.3)"),
                   ("value_classifier", "Classifier defence @F1"),
                   ("value_conservative", "Classifier defence @conservative"),
                   ("value_oracle", "Oracle defence (upper bound)")]:
        print("  " + summarise(lbl, col(k), unit="$"))

    print("\n-- Attack damage and recovery --")
    print("  " + summarise("Attack damage", col("damage_pct"), pct=True))
    for k, lbl in [("rec_penalised", "Recovered: blanket penalty"),
                   ("rec_classifier", "Recovered: classifier @F1"),
                   ("rec_conservative", "Recovered: classifier @conservative"),
                   ("rec_oracle", "Recovered: oracle (ceiling)")]:
        print("  " + summarise(lbl, col(k), pct=True))

    agg = {}
    for k in runs[0]:
        if k in ("importance", "seed"):
            continue
        agg[k] = {"mean": float(np.mean(col(k))), "std": float(np.std(col(k))), "values": col(k)}
    imp = {f: float(np.mean([r["importance"][f] for r in runs])) for f in runs[0]["importance"]}
    print("\n-- Mean feature importance (top 8) --")
    for f, v in sorted(imp.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {f:<18s} {v:.4f}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    out = args.out or os.path.join(ARTIFACT_DIR, f"results_{args.attack}.json")
    with open(out, "w") as fh:
        json.dump({"attack": args.attack, "seeds": args.seeds,
                   "aggregate": agg, "importance": imp, "runs": runs}, fh, indent=2, default=float)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
