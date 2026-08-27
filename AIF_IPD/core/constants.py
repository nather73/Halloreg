"""
core.constants
==============

Global constants, index conventions and the payoff-manipulation API
of the discrete Prisoner's Dilemma.

[State encoding]
One round of the simultaneous IPD is encoded as a 4-state **joint
outcome**, from the focal agent's viewpoint:

    state = 2 * (my action) + (their action)
      0 = CC   me C, them C   -> mutual cooperation
      1 = CD   me C, them D   -> I am the sucker
      2 = DC   me D, them C   -> I take the temptation
      3 = DD   me D, them D   -> mutual defection

[Payoffs]
Standard PD conditions T > R > P > S and 2R > T + S; defaults
R=3 (reward), T=5 (temptation), S=0 (sucker), P=1 (punishment).

[Non-stationary payoff support — H4/H4A/H5]
Varying the cooperation index CI = (R - P) / (T - S) per round changes
the game structure itself. `set_payoffs(ci)` keeps T=5, S=0, P=1 and
modulates only R = P + CI*(T - S), updating the module-global arrays
**in place** so already-imported references (PAYOFF_SELF etc.) see the
new payoffs immediately — the implementation basis of round-level
payoff variation.

    CI  < 0    : deadlock — mutual cooperation is worse than mutual
                 defection, so cooperating loses.
    CI = 0.4   : the default PD (R = 3).
    CI >= 1    : harmony — R > T, cooperation dominates.
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ Base payoffs
R, T, S, P = 3.0, 5.0, 0.0, 1.0

# ------------------------------------------------------------------ Index conventions
CC, CD, DC, DD = 0, 1, 2, 3
COOP, DEFECT = 0, 1

N_STATES = 4
N_ACTIONS = 2

ACTION_NAMES = {COOP: "C", DEFECT: "D"}
STATE_NAMES = {CC: "CC", CD: "CD", DC: "DC", DD: "DD"}

# joint outcome -> payoff, index order [CC, CD, DC, DD].
#   PAYOFF_SELF  : my payoff    (CC=R, CD=S, DC=T, DD=P)
#   PAYOFF_OTHER : their payoff (CC=R, CD=T, DC=S, DD=P) — symmetric
#                  game, only CD/DC swap.
PAYOFF_SELF = np.array([R, S, T, P], dtype=float)
PAYOFF_OTHER = np.array([R, T, S, P], dtype=float)

# (my action, their action) -> (my payoff, their payoff); used for
# perspective taking in EFE computations.
PD_PAYOFFS = {
    (COOP, COOP): (R, R),
    (COOP, DEFECT): (S, T),
    (DEFECT, COOP): (T, S),
    (DEFECT, DEFECT): (P, P),
}

# Cooperation index of the default PD = (3 - 1) / (5 - 0) = 0.4
DEFAULT_COOP_INDEX = round((R - P) / (T - S), 6)


# ------------------------------------------------------------------ Payoff setters
def set_payoffs(ci: float, strict: bool = False) -> tuple:
    """
    Reset payoffs from a cooperation index CI (the core hook of the
    non-stationary environment).

    Keeps T=5, S=0, P=1 and sets R = P + CI*(T - S) = 1 + 5*CI.
    strict=True enforces the valid-PD ordering T > R > P > S; the
    default strict=False admits out-of-PD structures (deadlock CI < 0,
    harmony CI >= 1) as members of non-stationary schedules.

    Updates the globals in place (slice assignment) so existing
    references stay live. Returns the updated (R, T, S, P).
    """
    global R, P
    ci = float(ci)
    R_new = P + ci * (T - S)
    if strict and not (T > R_new > P > S):
        raise ValueError(f"invalid cooperation index: {ci}")
    R = R_new
    PAYOFF_SELF[:] = [R, S, T, P]
    PAYOFF_OTHER[:] = [R, T, S, P]
    PD_PAYOFFS.update({(COOP, COOP): (R, R), (COOP, DEFECT): (S, T),
                       (DEFECT, COOP): (T, S), (DEFECT, DEFECT): (P, P)})
    return (R, T, S, P)


def set_payoff_matrix(R_new: float, T_new: float,
                      S_new: float, P_new: float) -> tuple:
    """
    General setter for a **direct** (R, T, S, P) — parameter-recovery
    (H6) designs only.

    Why it exists: the regressor of the opponent's empathy lambda_j in
    `empathy_shift` below has coefficient (T - S). `set_payoffs` fixes
    T and S, so that regressor is the constant 5 and becomes perfectly
    collinear with the intercept alpha — a rank-deficient design in
    which alpha and lambda_j cannot be identified separately. The H6
    recovery battery modulates (T, S) per block to centre the lambda_j
    regressor, hence this setter without the valid-PD constraint.

    Always return to the default PD with `reset_payoffs()` afterwards.
    """
    global R, T, S, P
    R, T, S, P = float(R_new), float(T_new), float(S_new), float(P_new)
    PAYOFF_SELF[:] = [R, S, T, P]
    PAYOFF_OTHER[:] = [R, T, S, P]
    PD_PAYOFFS.update({(COOP, COOP): (R, R), (COOP, DEFECT): (S, T),
                       (DEFECT, COOP): (T, S), (DEFECT, DEFECT): (P, P)})
    return (R, T, S, P)


def reset_payoffs() -> tuple:
    """Return to the default PD payoffs (R=3, T=5, S=0, P=1)."""
    return set_payoff_matrix(3.0, 5.0, 0.0, 1.0)


def current_payoffs() -> tuple:
    """Snapshot of the current (R, T, S, P) for workers/logging."""
    return (R, T, S, P)


# ------------------------------------------------------------------ Empathy shift
def empathy_shift(lam, my_coop_rate):
    """
    The payoff-derived part of the opponent's cooperate-defect utility
    gap dU = U(C) - U(D), on the logit scale, for an opponent with
    empathy weight lambda_j.

    Derivation. With my (focal) cooperation probability p, from the
    opponent's side:
        self part     dself(p) = p*(R - T) + (1 - p)*(S - P)
        empathy part  lambda_j * (T - S)
            (T - S) is the payoff range the opponent can hand me by
            cooperating; lambda_j weights that gain into their own
            utility.
    Collecting terms in p:
        empathy_shift(lam, p) = (T-S)*lam + (R-T+P-S)*p + (S-P)

    Default-PD coefficients: (5, -1, -1), i.e. 5*lam - p - 1. In the
    non-stationary environment the global R is read at call time, so
    the ToM likelihood automatically reflects the **current context's
    utilities** (context-dependent perspective taking).

    lam and my_coop_rate may be scalars or ndarrays (vectorised for
    the particle filter).
    """
    return (T - S) * lam + (R - T + P - S) * my_coop_rate + (S - P)


# ------------------------------------------------------------------ State transforms
def joint_index(my_act: int, opp_act: int) -> int:
    """(my action, their action) -> joint outcome index."""
    return int(my_act) * 2 + int(opp_act)


def split_joint(state: int) -> tuple[int, int]:
    """joint outcome -> (my action, their action)."""
    my_a, opp_a = divmod(int(state), 2)
    return my_a, opp_a


def opponent_action_from_state(state: int) -> int:
    """Extract the opponent action from a joint outcome."""
    return int(state) % 2


def my_action_from_state(state: int) -> int:
    """Extract my action from a joint outcome."""
    return int(state) // 2


def mirror_state(state):
    """
    Mirror a focal joint state into the **opponent's viewpoint**:
    (a_i, a_j) -> (a_j, a_i). Used in AIF-vs-AIF dyads so each agent
    receives observations from its own perspective.
    """
    if state is None:
        return None
    my_a, opp_a = split_joint(state)
    return joint_index(opp_a, my_a)
