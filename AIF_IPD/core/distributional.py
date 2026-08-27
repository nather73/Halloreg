"""
core.distributional
===================

**Quantile-grid utilities.**

`QuantileCode` (the marginal reward distribution) was retired in
v1.5.0: once both affect axes were grounded in the QRTD situational
values Z(s,a) (see core.core_affect), the marginal had no remaining
role.

  - valence : the rank of the current partner's expected-reward
              distribution (median) within the distance-weighted
              population of **other partners** held by SelfModel — a
              cross-sectional social comparison, not a self-comparison
              over time.
  - arousal : the update magnitude of Z(s,a) beliefs (W1 shift).

What remains is the grid definition, shared by core.qrtd and
core.self_model.
"""

from __future__ import annotations

import numpy as np


def midpoint_taus(n: int) -> tuple:
    """
    **Midpoint quantile grid** tau_i = (2i + 1) / (2n), i = 0..n-1.

    (1) The expectation reduces to a plain mean (midpoint rule).
    (2) Odd n places a knot exactly at tau = 0.5 — the basis of the
        valence sign identity. Hence the channel count must be odd.
    """
    if n % 2 == 0:
        raise ValueError("channel count must be odd (guarantees a tau=0.5 knot)")
    return tuple((2 * i + 1) / (2.0 * n) for i in range(n))


#: Default grid — 21 channels; the 11th (i = 10) is exactly 0.5.
DEFAULT_TAUS = midpoint_taus(21)


def vector_cdf(values: np.ndarray, taus: np.ndarray, x: float) -> float:
    """
    Cumulative probability F-hat(x) in (0, 1) of the distribution
    represented by a quantile vector. Linear interpolation inside the
    grid, exponential tails outside (decay length = distribution width
    — "the unit of surprise is the width of experience"); the open
    interval keeps valence off the boundary.
    """
    v = np.asarray(values, dtype=float)
    t = np.asarray(taus, dtype=float)
    s = float(v[-1] - v[0])
    if s < 1e-9:
        return 0.5
    x = float(x)
    eps = 1e-6
    if x <= v[0]:
        return float(np.clip(t[0] * np.exp(-(v[0] - x) / s), eps, 1 - eps))
    if x >= v[-1]:
        return float(np.clip(1.0 - (1.0 - t[-1]) * np.exp(-(x - v[-1]) / s),
                             eps, 1 - eps))
    k = int(np.searchsorted(v, x) - 1)
    k = max(0, min(k, len(v) - 2))
    lo, hi = v[k], v[k + 1]
    frac = 0.0 if hi - lo < 1e-12 else (x - lo) / (hi - lo)
    return float(t[k] + frac * (t[k + 1] - t[k]))
