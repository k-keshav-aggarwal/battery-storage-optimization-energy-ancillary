#!/usr/bin/env python3
"""Cross-family transfer: does a defence trained on one attack survive another?

Detectors and the classifier are fitted on the TRAIN+VAL window of a *source*
attack family, then applied unchanged to the TEST window of a *target* family
the defence has never seen. Reported numbers are all on the target family.

The ``--source`` flag accepts several families, so adversarial training
(fitting on more than one known attack) can be compared directly against
single-family training on the same held-out target.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from src.attack import ATTACKS
from src.classify import (auroc_on, conservative_threshold, select_detector_threshold,
                          train_classifier)
from src.config import ARTIFACT_DIR, SEEDS, VAE_EPOCHS
from src.data import load_hourly, split_masks
from src.detect import run_detectors
from src.dispatch import defend_by_price_correction, rolling_dispatch
from src.evaluate import classification_report_at, recovery_pct, summarise
from src.features import build_causal_features, expected_price_causal


def transfer_once(hourly, sources: list[str], target: str, seed: int,
                  epochs: int, flag_budget: float):
    """Fit on ``sources``, evaluate on ``target``, all on the same seed.

    Using one seed across families is deliberate: every family draws its event
    start times from the same generator before any shape-specific randomness,
    so for a given seed the spikes land in the *same hours* regardless of
    family. Differences between rows of the transfer matrix are therefore
    attributable to attack shape, not to where the attack happened to fall.
    """
    import pandas as pd

    from src.features import FEATURE_COLS

    dt, p_true = hourly["datetime"], hourly["SP15"].values
    tr, va, te = split_masks(hourly)

    # ---- source: fit detectors + classifier on train, threshold on val ----
    # Several source families stack as extra labelled rows: the defender knows
    # more than one attack, but still not the one it will face.
    feat_parts, y_parts, tr_parts, va_parts = [], [], [], []
    for si, src in enumerate(sources):
        atk_s = ATTACKS[src](p_true, seed=seed)
        det_s = run_detectors(dt, atk_s.observed, tr, seed=seed + si, epochs=epochs)
        thr_s = select_detector_threshold(det_s.ensemble, va, budget=flag_budget)
        flagged_s = det_s.ensemble >= thr_s
        feats_s = build_causal_features(dt, atk_s.observed, detector_scores=det_s.as_dict())

        keep = (tr | va) & flagged_s
        feat_parts.append(feats_s[keep])
        y_parts.append(atk_s.mask.astype(int)[keep])
        tr_parts.append(tr[keep])
        va_parts.append(va[keep])

    feats_fit = pd.concat(feat_parts, ignore_index=True)
    y_fit = np.concatenate(y_parts)
    fit_m = np.concatenate(tr_parts)          # train hours only
    val_m = np.concatenate(va_parts)          # validation hours only
    all_rows = np.ones(len(feats_fit), dtype=bool)

    clf = train_classifier(feats_fit, y_fit, all_rows, fit_mask=fit_m, val_mask=val_m, seed=seed)
    thr_cons = conservative_threshold(clf.p_synthetic, y_fit, all_rows, val_m,
                                      target_precision=0.99)

    # ---- target: apply the fitted defence unchanged ------------------------
    atk_t = ATTACKS[target](p_true, seed=seed)
    det_t = run_detectors(dt, atk_t.observed, tr, seed=seed, epochs=epochs)
    thr_t = select_detector_threshold(det_t.ensemble, va, budget=flag_budget)
    flagged_t = det_t.ensemble >= thr_t
    feats_t = build_causal_features(dt, atk_t.observed, detector_scores=det_t.as_dict())
    y_t = atk_t.mask.astype(int)

    p_t = np.zeros(len(feats_t))
    p_t[flagged_t] = clf.model.predict_proba(feats_t[FEATURE_COLS].values[flagged_t])[:, 1]

    expected = expected_price_causal(dt, atk_t.observed)
    pt, po, ex = p_true[te], atk_t.observed[te], expected[te]

    def value(price_decision):
        return rolling_dispatch(price_decision, pt).total_value

    v_clean, v_naive = value(pt), value(po)
    v_cons = value(defend_by_price_correction(po, ex, (p_t >= thr_cons)[te]))
    v_f1 = value(defend_by_price_correction(po, ex, (p_t >= clf.threshold)[te]))
    v_oracle = value(defend_by_price_correction(po, ex, atk_t.mask[te]))
    rep = classification_report_at(p_t, y_t, flagged_t, te, thr_cons)

    return {
        "seed": seed, "sources": "+".join(sources), "target": target,
        "auroc": auroc_on(p_t, y_t, flagged_t, te),
        "precision": rep["precision"], "recall": rep["recall"],
        "rec_conservative": recovery_pct(v_cons, v_naive, v_clean),
        "rec_f1": recovery_pct(v_f1, v_naive, v_clean),
        "rec_oracle": recovery_pct(v_oracle, v_naive, v_clean),
        "damage_pct": (v_naive - v_clean) / abs(v_clean) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    ap.add_argument("--flag-budget", type=float, default=0.12)
    ap.add_argument("--pairs", default="all",
                    help="'all' for the full matrix, or 'src1+src2>target' entries separated by commas")
    args = ap.parse_args()

    hourly = load_hourly()
    fams = list(ATTACKS)
    if args.pairs == "all":
        combos = [([s], t) for s, t in itertools.product(fams, fams)]
        # adversarial training: every pair of sources against the held-out third
        for t in fams:
            combos.append(([f for f in fams if f != t], t))
    else:
        combos = []
        for entry in args.pairs.split(","):
            s, t = entry.split(">")
            combos.append((s.split("+"), t.strip()))

    rows = []
    for sources, target in combos:
        runs = [transfer_once(hourly, sources, target, s, args.epochs, args.flag_budget)
                for s in args.seeds]
        rows.extend(runs)
        tag = f"{'+'.join(sources):>22s} -> {target:<11s}"
        held_out = target not in sources
        print(f"{tag} {'[held out]' if held_out else '[in-family]':12s} "
              f"AUROC={np.mean([r['auroc'] for r in runs]):.3f}  "
              f"prec={np.mean([r['precision'] for r in runs]):.3f}  "
              f"rec={np.mean([r['recall'] for r in runs]):.3f}  "
              f"recovered={np.mean([r['rec_conservative'] for r in runs]):+6.1f}%  "
              f"(ceiling {np.mean([r['rec_oracle'] for r in runs]):.1f}%)")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    out = os.path.join(ARTIFACT_DIR, "results_transfer.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
