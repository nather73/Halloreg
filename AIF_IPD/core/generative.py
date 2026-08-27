"""
core.generative
===============

Generative model (A, B, C, D) of the 4-state joint-outcome POMDP and
its **analytic EFE**.

[Why an analytic solution exists]
Albarracin et al. (2026) compute variational state inference and the
expected free energy with pymdp, but this task's generative model is
special in two ways:

    (1) A = I_4 — the joint outcome is observed directly (full
        observability), so the state posterior is the one-hot of the
        observation.
    (2) policy_len = 1 — one-step EFE; multi-step planning lives in a
        separate module.

Under these conditions the pymdp EFE equals the closed form below
**exactly**:

    neg_EFE(a) = E_{s'~B(.|a, pc)}[C[s']] + H[B(.|a, pc)]
                 (pragmatic value)          (state info gain:
                                             entropy, since A = I)
    G(a) = -neg_EFE(a)                     (lower = preferred)

where pc is the ToM-predicted opponent cooperation probability and the
transition B is

    a = C  ->  {CC: pc, CD: 1 - pc}
    a = D  ->  {DC: pc, DD: 1 - pc}.

The numpy solution is therefore an exact, orders-of-magnitude faster
replacement of the pymdp path, and expresses the recursive extensions
(the opponent's EFE, self-projection) cleanly.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    CC, CD, DC, DD, COOP, DEFECT,
    N_STATES, N_ACTIONS, PAYOFF_SELF, PAYOFF_OTHER,
)

_EPS = 1e-12


# ------------------------------------------------------------------ Generative model
def build_A() -> np.ndarray:
    """Observation likelihood A = I_4 (direct joint-outcome
    observation)."""
    return np.eye(N_STATES)


def build_B(prior_opp_coop: float = 0.5) -> np.ndarray:
    """
    Action-conditional transition B[s', s, a].

    Given my action a, the next outcome depends only on the opponent
    cooperation probability pc, not on the current state s (the
    simplification of Albarracin et al.). Returns shape (S', S, A).
    """
    pc = float(np.clip(prior_opp_coop, _EPS, 1 - _EPS))
    pd = 1.0 - pc
    B = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        B[CC, s, COOP] = pc
        B[CD, s, COOP] = pd
        B[DC, s, DEFECT] = pc
        B[DD, s, DEFECT] = pd
    return B


def build_C(perspective: str = "self") -> np.ndarray:
    """Log-preferences C; perspective in {'self', 'other'} —
    perspective taking swaps only C."""
    return PAYOFF_SELF.copy() if perspective == "self" else PAYOFF_OTHER.copy()


def build_D() -> np.ndarray:
    """Uniform initial state prior D."""
    return np.ones(N_STATES) / N_STATES


# ------------------------------------------------------------------ Analytic EFE
def predicted_next_state(pc: float) -> np.ndarray:
    """
    Predicted next-state distribution per action, shape
    (N_ACTIONS, N_STATES): row 0 = if I play C, row 1 = if I play D.
    """
    pc = float(np.clip(pc, _EPS, 1 - _EPS))
    pd = 1.0 - pc
    dist = np.zeros((N_ACTIONS, N_STATES))
    dist[COOP, CC] = pc
    dist[COOP, CD] = pd
    dist[DEFECT, DC] = pc
    dist[DEFECT, DD] = pd
    return dist


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy (natural log)."""
    p = np.clip(p, _EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def efe_terms(pc: float, C: np.ndarray) -> dict:
    """
    EFE decomposition per action under preferences C and opponent
    cooperation probability pc.

    Returns
      pragmatic : (A,)  E_{s'}[C[s']]        — expected utility
      epistemic : (A,)  H[B(.|a, pc)]        — state-uncertainty term
      neg_efe   : (A,)  pragmatic + epistemic (higher = preferred)
      G         : (A,)  -neg_efe             (lower = preferred)
    """
    dist = predicted_next_state(pc)                  # (2, 4)
    pragmatic = dist @ np.asarray(C, dtype=float)    # (2,)
    epistemic = np.array([_entropy(dist[a]) for a in range(N_ACTIONS)])
    neg_efe = pragmatic + epistemic
    return {"pragmatic": pragmatic, "epistemic": epistemic,
            "neg_efe": neg_efe, "G": -neg_efe}


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with a temperature parameter."""
    z = np.asarray(x, dtype=float) / max(temperature, _EPS)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)
