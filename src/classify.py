"""Genuine-vs-synthetic classification over detector-flagged hours.

Validation protocol
-------------------
The notebook lineage used ``StratifiedKFold(shuffle=True)`` over the flagged
hours. The 88 injected hours came from 16 *contiguous* events, so shuffling put
hours from the same spike in both the training and the held-out fold. With
smooth within-event features that is near-duplicate leakage, and it is the
third of three stacked leaks inflating the reported 0.9723 AUROC.

This module does not cross-validate at all. It fits on the train+validation
window and reports on a strictly later test window -- the same temporal
holdout the detectors and the dispatcher use. A ``GroupKFold`` helper keyed on
event id is provided for anyone who wants a CV estimate as well; it never
splits one event across folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:                                    # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_XGB = False


@dataclass
class ClassifierResult:
    p_synthetic: np.ndarray      # over every hour in the series (0 where not flagged)
    threshold: float             # chosen on validation
    model: object
    feature_importance: dict


def _make_model(seed: int, pos_weight: float):
    if _HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=seed, verbosity=0, n_jobs=2,
        )
    return HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                          learning_rate=0.05, random_state=seed)


def select_detector_threshold(scores: np.ndarray, val_mask: np.ndarray,
                              budget: float = 0.12) -> float:
    """Threshold that flags roughly ``budget`` of validation hours.

    Chosen on validation, never on test. The prefilter's job is recall; the
    classifier downstream supplies precision.
    """
    return float(np.quantile(scores[val_mask], 1.0 - budget))


def train_classifier(features, y_synthetic: np.ndarray, flagged: np.ndarray,
                     fit_mask: np.ndarray, val_mask: np.ndarray,
                     seed: int = 0) -> ClassifierResult:
    """Fit on flagged hours inside ``fit_mask``; pick the operating point on val.

    ``fit_mask`` should cover train (+ optionally validation). ``val_mask``
    selects the hours used to choose the decision threshold; it must be
    disjoint from the test window.
    """
    x_all = features[FEATURE_COLS].values
    fit_idx = np.where(fit_mask & flagged)[0]
    val_idx = np.where(val_mask & flagged)[0]

    y_fit = y_synthetic[fit_idx]
    if y_fit.sum() < 5 or (1 - y_fit).sum() < 5:
        raise ValueError(f"too few labelled examples to fit: {int(y_fit.sum())} synthetic")

    pos_frac = max(y_fit.mean(), 1e-6)
    model = _make_model(seed, pos_weight=(1 - pos_frac) / pos_frac)
    model.fit(x_all[fit_idx], y_fit)

    p_all = np.zeros(len(features))
    p_all[flagged] = model.predict_proba(x_all[flagged])[:, 1]

    # Operating point: best F1 on validation-flagged hours.
    y_val, p_val = y_synthetic[val_idx], p_all[val_idx]
    if y_val.sum() >= 2 and (1 - y_val).sum() >= 2:
        prec, rec, thr = precision_recall_curve(y_val, p_val)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        threshold = float(thr[max(0, int(np.argmax(f1)) - 1)]) if len(thr) else 0.5
    else:
        threshold = 0.5

    if _HAS_XGB:
        imp = dict(zip(FEATURE_COLS, model.feature_importances_.astype(float)))
    else:
        imp = {c: float("nan") for c in FEATURE_COLS}

    return ClassifierResult(p_synthetic=p_all, threshold=threshold, model=model,
                            feature_importance=dict(sorted(imp.items(), key=lambda kv: -kv[1])))


def conservative_threshold(p_synthetic: np.ndarray, y_synthetic: np.ndarray,
                           flagged: np.ndarray, val_mask: np.ndarray,
                           target_precision: float = 0.99) -> float:
    """Highest-precision usable operating point, chosen on validation.

    Walks the validation precision-recall curve and takes the threshold with
    the greatest recall whose precision still meets ``target_precision``. If
    nothing reaches that bar, falls back to the most precise point that still
    flags at least one hour -- returning a threshold that flags nothing would
    make the defence a no-op and hide the failure.

    This regime is the economically important one: false positives land on
    genuine high-price hours, which are exactly the hours worth selling into.
    """
    idx = np.where(val_mask & flagged)[0]
    if len(idx) == 0:
        return 1.0
    y, p = y_synthetic[idx], p_synthetic[idx]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return 0.5

    prec, rec, thr = precision_recall_curve(y, p)
    # precision_recall_curve returns one more point than thresholds
    prec, rec = prec[:-1], rec[:-1]
    ok = (prec >= target_precision) & (rec > 0)
    if ok.any():
        return float(thr[np.argmax(np.where(ok, rec, -1.0))])
    viable = rec > 0
    return float(thr[np.argmax(np.where(viable, prec, -1.0))]) if viable.any() else 0.5


def auroc_on(p_synthetic: np.ndarray, y_synthetic: np.ndarray,
             flagged: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask & flagged)[0]
    y = y_synthetic[idx]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y, p_synthetic[idx]))
