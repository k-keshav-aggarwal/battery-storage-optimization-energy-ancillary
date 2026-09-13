"""Guardrail tests for the properties that were violated in the notebook lineage.

Run with ``python3 -m pytest tests/ -q`` or directly: ``python3 tests/test_pipeline.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attack import ATTACKS, inject_naive
from src.config import CAPACITY_MWH, EFFICIENCY_ONE_WAY, MAX_CHARGE_MW, MAX_DISCHARGE_MW
from src.data import load_hourly, split_masks
from src.detect import TrainStats
from src.dispatch import defend_by_price_correction, rolling_dispatch
from src.features import FEATURE_COLS, build_causal_features, expected_price_causal

_HOURLY = None


def hourly():
    global _HOURLY
    if _HOURLY is None:
        _HOURLY = load_hourly()
    return _HOURLY


def test_features_are_strictly_causal():
    """No feature at t may respond to any price after t.

    The original pipeline's three highest-importance features all used
    ``center=True`` rolling or ``shift(-window)``, so this test would have
    failed on it.
    """
    h = hourly()
    cut = 5000
    p = h.SP15.values.copy()
    scrambled = p.copy()
    scrambled[cut:] = scrambled[cut:] * 7.5 + 1000.0

    a = build_causal_features(h.datetime, p)[FEATURE_COLS].values[:cut]
    b = build_causal_features(h.datetime, scrambled)[FEATURE_COLS].values[:cut]
    assert np.abs(a - b).max() == 0.0, "feature leaked information from the future"

    ea = expected_price_causal(h.datetime, p)[:cut]
    eb = expected_price_causal(h.datetime, scrambled)[:cut]
    assert np.abs(ea - eb).max() == 0.0, "expected price leaked information from the future"


def test_attack_leaves_settlement_and_unflagged_hours_untouched():
    p = hourly().SP15.values
    for name, fn in ATTACKS.items():
        a = fn(p, seed=3)
        assert np.array_equal(a.settlement, p), f"{name} modified the settlement series"
        assert np.array_equal(a.observed[~a.mask], p[~a.mask]), f"{name} touched unflagged hours"
        assert a.mask.sum() > 0, f"{name} injected nothing"
        assert (a.event_id[a.mask] >= 0).all(), f"{name} left an injected hour without an event id"
        assert (a.event_id[~a.mask] == -1).all(), f"{name} tagged a clean hour with an event id"


def test_dispatch_respects_battery_physics():
    p = hourly().SP15.values[:2400]
    r = rolling_dispatch(p, p)
    tol = 1e-6

    assert r.charge.min() >= -tol and r.charge.max() <= MAX_CHARGE_MW + tol
    assert r.discharge.min() >= -tol and r.discharge.max() <= MAX_DISCHARGE_MW + tol
    assert r.soc.min() >= -tol and r.soc.max() <= CAPACITY_MWH + tol

    # SOC must follow its own dynamics, not drift.
    step = r.charge * EFFICIENCY_ONE_WAY - r.discharge / EFFICIENCY_ONE_WAY
    expected = CAPACITY_MWH * 0.5 + np.cumsum(step)
    assert np.abs(r.soc - expected).max() < 1e-5, "SOC trace does not match the dynamics"

    # Fees on both legs make simultaneous charge+discharge strictly wasteful,
    # so it should never appear even though no binary variable forbids it.
    both = (r.charge > 1e-6) & (r.discharge > 1e-6)
    assert not both.any(), f"{both.sum()} hours charge and discharge at once"


def test_constrained_defence_cannot_beat_the_unattacked_optimum():
    """The invariant the original pipeline reported +7.70% against.

    Any defence only ever alters the prices the optimiser believes; it cannot
    make the battery earn more than dispatching on the true schedule does. A
    result claiming otherwise is reading a reweighted objective, not profit.
    """
    h = hourly()
    p_true = h.SP15.values[:2400]
    a = inject_naive(h.SP15.values, seed=1)
    p_obs = a.observed[:2400]
    mask = a.mask[:2400]
    exp = expected_price_causal(h.datetime[:2400], p_obs)

    clean = rolling_dispatch(p_true, p_true).total_value
    for flags in (mask, np.ones_like(mask), np.zeros_like(mask)):
        v = rolling_dispatch(defend_by_price_correction(p_obs, exp, flags), p_true).total_value
        assert v <= clean + 1e-6, "defence beat the unattacked optimum -- profit accounting is wrong"


def test_attack_is_costly_and_oracle_defence_recovers_most_of_it():
    h = hourly()
    _, _, te = split_masks(h)
    p_true = h.SP15.values
    a = inject_naive(p_true, seed=0)
    exp = expected_price_causal(h.datetime, a.observed)

    pt, po = p_true[te], a.observed[te]
    clean = rolling_dispatch(pt, pt).total_value
    naive = rolling_dispatch(po, pt).total_value
    oracle = rolling_dispatch(defend_by_price_correction(po, exp[te], a.mask[te]), pt).total_value

    assert naive < clean, "attack did not cost the undefended dispatcher anything"
    damage = clean - naive
    assert damage / clean > 0.02, f"attack damage {damage/clean:.3%} too small to measure a defence against"
    assert (oracle - naive) / damage > 0.5, "oracle defence failed to recover most of the damage"


def test_train_stats_use_only_training_data():
    """Score normalisation must not be rescaled by the evaluation window."""
    train = np.random.default_rng(0).normal(size=5000)
    stats = TrainStats.fit(train)
    lo, hi = stats.lo, stats.hi

    huge = np.concatenate([train, np.array([1e9, -1e9])])
    assert TrainStats.fit(train).lo == lo and TrainStats.fit(train).hi == hi
    # Applying to out-of-range values clips rather than rescaling.
    out = stats.apply(huge)
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_splits_are_disjoint_and_ordered():
    h = hourly()
    tr, va, te = split_masks(h)
    assert (tr & va).sum() == 0 and (va & te).sum() == 0 and (tr & te).sum() == 0
    assert tr.sum() and va.sum() and te.sum()
    assert h.datetime[tr].max() < h.datetime[va].min() < h.datetime[te].min()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
