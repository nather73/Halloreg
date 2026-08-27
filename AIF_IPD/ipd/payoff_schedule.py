"""
ipd.payoff_schedule
===================

**Non-stationary payoff schedules** — the environment of H4/H4A/H5.

[Motivation]
Fixed strategies (TFT/GTFT/WSLS/ALLC/ALLD) **never read the payoff
matrix**; they react only to action history. Active-inference agents
compute the EFE preference C from the current payoffs and can
re-evaluate self/other utilities context-dependently as the game
structure shifts. The claim "under non-stationary payoffs it achieves
higher final payoffs than static strategies" rests on this asymmetry.

[Operationalisation]
The discrete IPD is kept while the **cooperation index**
CI = (R - P) / (T - S) varies per round: T=5, S=0, P=1 fixed and
R = P + CI*(T - S) = 1 + 5*CI.

    CI = 0    deadlock boundary : R = P — mutual cooperation's gain
                                  vanishes (schedules are limited to
                                  CI in [0, 1] since v3.9.0).
    CI = 0.2  harsh PD          : 2R < T + S — cooperation fragile.
    CI = 0.4  the default PD
    CI = 0.8  generous PD       : still T > R, strong incentive.
    CI = 1    harmony           : R = 6 > T — cooperation dominates.

[Regime catalogue]
Six regimes with qualitatively different variation, each a
(t, T) -> CI function:

  stationary : control — fixed PD (CI = 0.4), same structure as H3.
  blocks     : staircase block switches (harmony -> PD -> deadlock ->
               PD -> harmony); adaptation to abrupt changes.
  oscillate  : sinusoidal oscillation; tracking of continuous
               periodic variation.
  aba        : A -> B -> A return; reversal learning on context
               return.
  drift      : deterministic linear drift; slow monotone change.
  shock      : mostly the default PD with a sharp deadlock shock in
               one window; self-protection under rare acute crisis.

Regime functions are **deterministic** (functions of the round index
alone), so within a regime every type and seed experiences the same
payoff trajectory and comparisons are paired.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

CIFn = Callable[[int, int], float]


# ------------------------------------------------------------------ Primitives
def ci_constant(ci: float) -> CIFn:
    """Constant CI on every round."""
    return lambda t, T: float(ci)


def ci_blocks(values) -> CIFn:
    """Equal blocks over the horizon, one CI each (staircase)."""
    vals = [float(v) for v in values]

    def f(t: int, T: int) -> float:
        k = len(vals)
        b = min(k - 1, int(t * k / max(T, 1)))
        return vals[b]
    return f


def ci_oscillate(lo: float, hi: float, period: int) -> CIFn:
    """Sinusoidal CI between lo and hi with the given period."""
    def f(t: int, T: int) -> float:
        phase = 0.5 * (1.0 + np.sin(2.0 * np.pi * t / max(period, 1)))
        return float(lo + (hi - lo) * phase)
    return f


def ci_aba(a: float, b: float) -> CIFn:
    """Three phases A -> B -> A (context switch and return)."""
    def f(t: int, T: int) -> float:
        third = max(T // 3, 1)
        return float(a if (t < third or t >= 2 * third) else b)
    return f


def ci_drift(start: float, end: float) -> CIFn:
    """Linear drift from start to end (slow monotone change)."""
    def f(t: int, T: int) -> float:
        frac = t / max(T - 1, 1)
        return float(start + (end - start) * frac)
    return f


def ci_shock(base: float, shock: float,
             onset_frac: float = 0.4, width_frac: float = 0.15) -> CIFn:
    """Hold the base CI except for one sharp shock window."""
    def f(t: int, T: int) -> float:
        lo = int(onset_frac * T)
        hi = int((onset_frac + width_frac) * T)
        return float(shock if lo <= t < hi else base)
    return f


# ------------------------------------------------------------------ Catalogue
#: Name -> CI function. The experiment CLI and workers refer to
#: regimes by name only (function objects are awkward to pickle under
#: spawn, so **strings are passed**).
#: REGIME_PERIOD: the repetition period of a pattern in rounds. The
#: pattern is defined on a **fixed period**, repeated as t mod PERIOD
#: (v3.7) rather than stretched over the horizon T: stretching would
#: let the "train 600R, evaluate 601-800R" protocol see only the tail
#: of the pattern, whereas periodic repetition puts exactly one full
#: cycle in the 200R evaluation window and gives the training span
#: three passes over the same pattern — measuring *learned*
#: non-stationary adaptation.
REGIME_PERIOD: int = 200


def ci_periodic(fn: CIFn, period: int = REGIME_PERIOD) -> CIFn:
    """Wrap (t, T) -> (t mod period, period) to repeat the pattern
    periodically."""
    def f(t: int, T: int) -> float:
        return fn(int(t) % int(period), int(period))
    return f


CI_REGIMES: Dict[str, CIFn] = {
    "stationary": ci_constant(0.4),
    "blocks": ci_periodic(ci_blocks([1.0, 0.4, 0.0, 0.4, 1.0])),
    "oscillate": ci_oscillate(0.0, 1.0, period=30),    # already 30R-periodic
    "aba": ci_periodic(ci_aba(1.0, 0.0)),
    "drift": ci_periodic(ci_drift(1.0, 0.0)),
    "shock": ci_periodic(ci_shock(0.4, 0.0, onset_frac=0.40,
                                  width_frac=0.15)),
}

#: Non-stationary regimes used in H4/H4A/H5 (stationary excluded).
NONSTATIONARY_REGIMES: List[str] = ["blocks", "oscillate", "aba",
                                    "drift", "shock"]

#: Display labels (identifier name kept for call-site
#: compatibility; values are English).
REGIME_LABEL_KO = {
    "stationary": "stationary (fixed PD)",
    "blocks": "block switches",
    "oscillate": "sinusoidal",
    "aba": "A-B-A return",
    "drift": "linear drift",
    "shock": "acute shock",
}


def _clip01(fn: CIFn) -> CIFn:
    """[v3.9.0] Schedule invariant — CI in [0, 1]. The catalogue is
    already within [0, 1]; this is a defensive clip for arbitrary
    functions and composition mistakes."""
    def f(t: int, T: int) -> float:
        return float(min(1.0, max(0.0, fn(t, T))))
    return f


def get_regime(name: str) -> CIFn:
    """Regime name -> CI function (workers rebuild from strings)."""
    if name not in CI_REGIMES:
        raise KeyError(f"unknown payoff regime: {name}")
    return _clip01(CI_REGIMES[name])


def regime_trace(name: str, n_rounds: int = 120) -> np.ndarray:
    """CI trajectory of a regime (for visualisation)."""
    f = get_regime(name)
    return np.array([f(t, n_rounds) for t in range(n_rounds)])
