"""Price-manipulation attacks on the announced day-ahead schedule.

Threat model
------------
The operator receives a day-ahead price schedule and dispatches against it.
An adversary with write access to that feed (a compromised market-data relay,
a spoofed vendor API, or a manipulated forecast service) injects artificial
spikes. Settlement happens at the *true* cleared price, so trusting an
injected spike is directly costly: the battery discharges into a price that
was never there.

Every injector returns the observed (attacked) series, the untouched
settlement series, an hour-level boolean mask, and an event id per hour so
that cross-validation can group by event instead of splitting one spike
across folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    NON_REVERT_FRACTION,
    SPIKE_DURATION_HOURS,
    SPIKE_INTERVAL_HOURS,
    SPIKE_MAGNITUDE,
)


@dataclass
class AttackedSeries:
    """One attacked price series plus everything needed to score it."""

    observed: np.ndarray      # what the dispatcher sees
    settlement: np.ndarray    # what it actually gets paid
    mask: np.ndarray          # True where an hour was injected
    event_id: np.ndarray      # -1 for untouched hours, else the event index

    @property
    def n_events(self) -> int:
        return int(self.event_id.max()) + 1 if (self.event_id >= 0).any() else 0

    def as_frame(self, datetimes) -> pd.DataFrame:
        return pd.DataFrame({
            "datetime": datetimes,
            "price_observed": self.observed,
            "price_settlement": self.settlement,
            "is_synthetic": self.mask.astype(int),
            "event_id": self.event_id,
        })


def _event_starts(n: int, interval: int, duration: int, rng: np.random.Generator) -> list[int]:
    """Jittered start indices, spaced so events never overlap."""
    starts, cursor = [], int(rng.integers(0, interval))
    while cursor + duration + 4 < n:
        starts.append(cursor)
        gap = int(rng.integers(int(interval * 0.5), int(interval * 1.5)))
        cursor += max(duration + 4, gap)
    return starts


def _finalise(price: np.ndarray, base: np.ndarray, hours, ids) -> AttackedSeries:
    mask = np.zeros(len(base), dtype=bool)
    event_id = np.full(len(base), -1, dtype=int)
    for h, i in zip(hours, ids):
        mask[h] = True
        event_id[h] = i
    return AttackedSeries(observed=price, settlement=base.copy(), mask=mask, event_id=event_id)


def inject_naive(prices: np.ndarray, seed: int = 0,
                 interval: int = SPIKE_INTERVAL_HOURS,
                 duration: int = SPIKE_DURATION_HOURS,
                 magnitude: float = SPIKE_MAGNITUDE,
                 non_revert_fraction: float = NON_REVERT_FRACTION) -> AttackedSeries:
    """Sharp multiplicative spikes; most plateau rather than revert.

    This is the original attack family from the notebook lineage, restated
    on a clean hourly series.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(prices, dtype=float)
    price = base.copy()
    n = len(base)
    median = float(np.median(base))

    hours, ids = [], []
    for idx, start in enumerate(_event_starts(n, interval, duration, rng)):
        dur = int(rng.integers(2, max(3, duration + 1)))
        end = min(start + dur, n - 1)
        mag = float(np.clip(magnitude + rng.normal(0, 0.1 * magnitude), 1.2, magnitude * 1.1))

        if rng.random() > non_revert_fraction:                  # reverting
            price[start:end + 1] = base[start:end + 1] * mag
        else:                                                   # plateau / slow decay
            peak = base[start] * mag
            rise = max(2, (end - start + 1) // 2)
            rise_end = min(start + rise - 1, end)
            price[start:rise_end + 1] = np.linspace(base[start], peak, rise_end - start + 1)
            if start + rise <= end:
                final = median * float(rng.uniform(0.60, 0.95))
                price[start + rise:end + 1] = np.linspace(peak, final, end - (start + rise) + 1)

        hours.extend(range(start, end + 1))
        ids.extend([idx] * (end - start + 1))

    return _finalise(price, base, hours, ids)


def inject_adaptive(prices: np.ndarray, seed: int = 0,
                    interval: int = SPIKE_INTERVAL_HOURS,
                    duration: int = SPIKE_DURATION_HOURS,
                    magnitude: float = SPIKE_MAGNITUDE) -> AttackedSeries:
    """Evasion-aware attack: gradual ramps that mimic genuine congestion.

    Built to defeat rate-of-change and smoothness detectors -- the rise is
    spread over the whole event and blended into the surrounding level, so no
    single hour shows a large jump.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(prices, dtype=float)
    price = base.copy()
    n = len(base)

    hours, ids = [], []
    for idx, start in enumerate(_event_starts(n, interval, duration, rng)):
        dur = int(rng.integers(max(4, duration - 2), duration + 3))
        end = min(start + dur, n - 1)
        span = end - start + 1
        if span < 3:
            continue
        mag = float(np.clip(magnitude * rng.uniform(0.8, 1.1), 1.2, magnitude * 1.2))

        # Smooth raised-cosine bump: continuous in value and first derivative.
        t = np.linspace(0.0, np.pi, span)
        bump = (1.0 - np.cos(2 * t)) / 2.0 if span > 3 else np.sin(t)
        envelope = 1.0 + (mag - 1.0) * bump
        price[start:end + 1] = base[start:end + 1] * envelope

        hours.extend(range(start, end + 1))
        ids.extend([idx] * span)

    return _finalise(price, base, hours, ids)


def inject_sinusoidal(prices: np.ndarray, seed: int = 0,
                      interval: int = SPIKE_INTERVAL_HOURS,
                      duration: int = SPIKE_DURATION_HOURS,
                      magnitude: float = SPIKE_MAGNITUDE) -> AttackedSeries:
    """A structurally different family, held out to test transfer.

    Multi-cycle oscillation rather than a single bump: a defence that has
    only ever seen ramps has not seen this shape.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(prices, dtype=float)
    price = base.copy()
    n = len(base)

    hours, ids = [], []
    for idx, start in enumerate(_event_starts(n, interval, duration, rng)):
        dur = int(rng.integers(max(5, duration - 1), duration + 4))
        end = min(start + dur, n - 1)
        span = end - start + 1
        if span < 4:
            continue
        mag = float(np.clip(magnitude * rng.uniform(0.8, 1.1), 1.2, magnitude * 1.2))
        cycles = float(rng.uniform(1.5, 2.5))
        phase = float(rng.uniform(0, np.pi))

        osc = np.sin(np.linspace(0, cycles * 2 * np.pi, span) + phase)
        taper = np.sin(np.linspace(0, np.pi, span))          # fade in/out at the edges
        price[start:end + 1] = base[start:end + 1] * (1.0 + (mag - 1.0) * osc * taper)

        hours.extend(range(start, end + 1))
        ids.extend([idx] * span)

    return _finalise(price, base, hours, ids)


ATTACKS = {
    "naive": inject_naive,
    "adaptive": inject_adaptive,
    "sinusoidal": inject_sinusoidal,
}
