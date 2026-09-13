"""Day-ahead rolling-horizon battery dispatch.

The notebook lineage solved a single linear program over the entire multi-year
price series. That is clairvoyant: an optimiser that already knows every future
price is close to immune to a manipulated one, which is why the measured attack
damage there was only -0.46% and no defence could recover more than a rounding
error.

Here the dispatcher instead does what a CAISO participant does: once per day it
receives a 24-hour announced price schedule and optimises against it, carrying
state of charge across days. A 24-hour horizon on announced day-ahead prices is
the market design, not foresight -- and it means a corrupted schedule leads
directly to a bad commitment that is then settled at the true cleared price.

Solved with ``scipy.optimize.linprog(method="highs")`` -- the same HiGHS solver
the original used through Pyomo, without the per-model construction overhead
that makes hundreds of sequential solves impractical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .config import (
    CAPACITY_MWH,
    DA_HORIZON_HOURS,
    DEGRADATION_COST,
    EFFICIENCY_ONE_WAY,
    INITIAL_SOC_FRAC,
    MAX_CHARGE_MW,
    MAX_DISCHARGE_MW,
    TRANSACTION_FEE,
)

UNIT_COST = TRANSACTION_FEE + DEGRADATION_COST


@dataclass
class DispatchResult:
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    cash_profit: float          # settled at true prices
    terminal_value: float       # leftover energy, valued identically for every strategy
    throughput_mwh: float

    @property
    def total_value(self) -> float:
        return self.cash_profit + self.terminal_value

    @property
    def equivalent_full_cycles(self) -> float:
        return self.throughput_mwh / CAPACITY_MWH


def _solve_block(prices: np.ndarray, soc0: float, terminal_price: float,
                 max_discharge: np.ndarray, capacity: float = CAPACITY_MWH):
    """One horizon LP. Returns (charge, discharge) in MW per hour.

    Variables are [charge_0..charge_{H-1}, discharge_0..discharge_{H-1}].
    State of charge is substituted out, so the only constraints are the running
    energy bounds -- a dense lower-triangular pair of blocks.

    Simultaneous charge and discharge is never optimal while ``UNIT_COST > 0``
    (both legs pay it), so it needs no binary variable to exclude.
    """
    h = len(prices)
    e = EFFICIENCY_ONE_WAY

    # soc_k = soc0 + sum_{j<=k} (e * charge_j - discharge_j / e)
    tri = np.tril(np.ones((h, h)))
    a_up = np.hstack([e * tri, -tri / e])        #  soc_k - soc0 <= capacity - soc0
    a_lo = -a_up                                  # -(soc_k - soc0) <= soc0
    a_ub = np.vstack([a_up, a_lo])
    b_ub = np.concatenate([np.full(h, capacity - soc0), np.full(h, soc0)])

    # Leftover energy is worth what it could be sold for, so the last hour of
    # each block is not treated as a deadline to empty the battery.
    tv = terminal_price * e
    cost_charge = (prices + UNIT_COST) - tv * e
    cost_discharge = -(prices - UNIT_COST) + tv / e

    res = linprog(
        c=np.concatenate([cost_charge, cost_discharge]),
        A_ub=a_ub, b_ub=b_ub,
        bounds=[(0, MAX_CHARGE_MW)] * h + [(0, float(m)) for m in max_discharge],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:h], res.x[h:]


def rolling_dispatch(price_decision: np.ndarray,
                     price_settlement: np.ndarray,
                     discharge_limit: np.ndarray | None = None,
                     horizon: int = DA_HORIZON_HOURS,
                     capacity: float = CAPACITY_MWH,
                     initial_soc_frac: float = INITIAL_SOC_FRAC) -> DispatchResult:
    """Roll a ``horizon``-hour LP across the window, one block per market day.

    ``price_decision`` is the schedule the operator believes (post-attack,
    post-defence). ``price_settlement`` is what it is actually paid. They are
    identical only when there is no attack.
    """
    n = len(price_decision)
    if len(price_settlement) != n:
        raise ValueError("decision and settlement series must be the same length")
    if discharge_limit is None:
        discharge_limit = np.full(n, MAX_DISCHARGE_MW)

    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc_trace = np.zeros(n)
    soc = capacity * initial_soc_frac

    # Reference price for valuing carry-over energy: the trailing level of the
    # believed schedule, so it stays available at decision time.
    ref = float(np.median(price_decision[:horizon]))

    for start in range(0, n, horizon):
        end = min(start + horizon, n)
        block = price_decision[start:end]
        c_blk, d_blk = _solve_block(block, soc, ref, discharge_limit[start:end], capacity)

        charge[start:end] = c_blk
        discharge[start:end] = d_blk
        for k in range(end - start):
            soc += c_blk[k] * EFFICIENCY_ONE_WAY - d_blk[k] / EFFICIENCY_ONE_WAY
            soc = min(max(soc, 0.0), capacity)
            soc_trace[start + k] = soc

        # Update the carry-over reference from prices already seen.
        ref = float(np.median(price_decision[max(0, end - 168):end]))

    cash = float(np.sum(discharge * (price_settlement - UNIT_COST)
                        - charge * (price_settlement + UNIT_COST)))
    terminal = float(soc * np.mean(price_settlement) * EFFICIENCY_ONE_WAY)
    throughput = float(np.sum(discharge))

    return DispatchResult(charge=charge, discharge=discharge, soc=soc_trace,
                          cash_profit=cash, terminal_value=terminal,
                          throughput_mwh=throughput)


# --------------------------------------------------------------- defences ---

def defend_by_price_correction(price_observed: np.ndarray,
                               price_expected: np.ndarray,
                               flagged: np.ndarray,
                               strength: float = 1.0) -> np.ndarray:
    """Distrust the printed price on flagged hours; fall back to the forecast.

    ``strength`` blends between believing the schedule (0) and fully replacing
    a flagged hour with the model's expected price (1).
    """
    corrected = price_observed.copy()
    idx = np.asarray(flagged, dtype=bool)
    corrected[idx] = ((1 - strength) * price_observed[idx]
                      + strength * price_expected[idx])
    return corrected


def defend_by_clipping(anomaly: np.ndarray, lam: float = 0.3) -> np.ndarray:
    """Blanket exposure limit: shrink the discharge cap by the anomaly score.

    This is the original 'penalised' regime, kept as a comparison point.
    """
    return np.clip(MAX_DISCHARGE_MW * (1.0 - lam * np.clip(anomaly, 0, 1)), 0.0, MAX_DISCHARGE_MW)
