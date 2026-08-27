"""
ipd.env
=======

The IPD environment and fixed-strategy opponents (generative
processes).

[Strategy set]
Five of the six types named by H1/H3/H4/H5 are fixed strategies:

  TFT   (tit_for_tat)  : return the opponent's last action.
  GTFT  (generous_tft) : TFT, but forgive a defection with
                         probability `generosity` (Nowak & Sigmund
                         1992).
  WSLS  (wsls)         : Win-Stay Lose-Shift — keep the own action
                         after a "win" (mutual cooperation CC or
                         successful exploitation DC), switch
                         otherwise.
  ALLC  (allc)         : always cooperate.
  ALLD  (alld)         : always defect — the intentional exploiter.

The sixth type, HalloReg, is ipd.agent.HalloRegAgent.

[Execution noise vs intent — the design basis of H2A]
`error` is the probability that the **emitted action is the opposite
of the intended one** — an operationalisation of low behavioural
precision beta, a different level from trait (alpha/rho/lambda_j)
change. This separation is what makes the H2A condition possible:
an intentional exploiter (ALLD, error = 0) and a noisy cooperator
(noisy TFT, error = 0.2) produce **the same defection observations**
from different latent causes.

[Trait-switch schedule — the design basis of H1A]
With `schedule = [(round, kind), ...]` the **same individual's
disposition changes** at the given rounds (a change of intent, not
noise). The memory-1 internal state carries across phases, so the
semantics is "the same person changed", not "a replacement".
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import COOP, DEFECT, joint_index

# Canonical names and display order of the five fixed strategies
# used in the population experiments (H3-H5).
FIXED_STRATEGIES = ("tft", "gtft", "wsls", "allc", "alld")
# Full type order including HalloReg — matches the row/column order
# of the payoff matrix Pi.
ALL_TYPES = FIXED_STRATEGIES + ("halloreg",)

# Display labels for figures and tables (identifier name kept for
# call-site compatibility; values are English).
TYPE_LABEL_KO = {
    "tft": "TFT", "gtft": "GTFT", "wsls": "WSLS",
    "allc": "ALLC", "alld": "ALLD", "halloreg": "HalloReg",
}


class Environment:
    """Simultaneous IPD environment: maps two actions to a joint
    outcome."""

    @staticmethod
    def step(my_action: int, opp_action: int) -> int:
        return joint_index(my_action, opp_action)


class StrategyAgent:
    """
    Fixed-strategy agent with the `act()` / `observe(focal_action)`
    interface.

    Parameters
    ----------
    kind : str
        One of {'tft', 'gtft', 'wsls', 'allc', 'alld', 'random'}.
    error : float
        Execution-noise rate — probability of emitting the opposite
        of the intended action.
    generosity : float
        GTFT's forgiveness probability (cooperate despite a
        defection).
    schedule : [(round, kind), ...] | None
        Trait-switch schedule; kind changes from the given round.
    """

    def __init__(self, kind: str = "tft", error: float = 0.0,
                 generosity: float = 0.3, seed: int = 0,
                 schedule: Optional[List[Tuple[int, str]]] = None):
        self.kind = kind
        self.base_kind = kind
        self.schedule = sorted(schedule or [], key=lambda x: x[0])
        self.error = float(error)
        self.generosity = float(generosity)
        self.rng = np.random.default_rng(seed)

        # memory-1 state: own last action and the observed focal
        # last action. Both start at cooperation by convention, so
        # TFT cooperates in round 1.
        self.my_last = COOP
        self.other_last = COOP
        self.round = 0

    # ------------------------------------------------------------ Intent
    def _intended(self) -> int:
        """The intended action **before** execution noise."""
        k = self.kind
        if k == "allc":
            return COOP
        if k == "alld":
            return DEFECT
        if k == "tft":
            return self.other_last
        if k == "gtft":
            # Forgive (cooperate) with probability generosity even
            # after a defection
            if self.other_last == DEFECT and self.rng.random() < self.generosity:
                return COOP
            return self.other_last
        if k == "wsls":
            # "Win" = the opponent cooperated (CC if I played C, a
            # successful DC exploitation if I played D). Stay after a
            # win, shift after a loss.
            win = (self.other_last == COOP)
            return self.my_last if win else (1 - self.my_last)
        if k == "random":
            return COOP if self.rng.random() < 0.5 else DEFECT
        return COOP

    # ------------------------------------------------------------ Emission
    def act(self) -> int:
        """The action actually emitted this round."""
        # Apply the trait-switch schedule (cumulative — the last
        # reached phase wins)
        for r, k in self.schedule:
            if self.round >= r:
                self.kind = k
        self.round += 1

        a = self._intended()
        if self.error > 0.0 and self.rng.random() < self.error:
            a = 1 - a               # execution noise: emit the opposite
        self.my_last = a
        return a

    def observe(self, focal_action: int) -> None:
        """Observe the action the focal agent actually emitted."""
        self.other_last = int(focal_action)

    def begin_partner(self, identity: int) -> None:
        """Interface compatibility (fixed strategies keep no
        identity memory)."""
        return None


# ------------------------------------------------------------------ Factory
_PRESETS = {
    "tft": dict(kind="tft", error=0.0),
    "gtft": dict(kind="gtft", error=0.0, generosity=0.3),
    "wsls": dict(kind="wsls", error=0.0),
    "allc": dict(kind="allc", error=0.0),
    "alld": dict(kind="alld", error=0.0),
    "random": dict(kind="random", error=0.0),
    # H2/H2A-specific conditions
    "exploiter": dict(kind="alld", error=0.0),        # intentional exploiter
    "noisy_tft": dict(kind="tft", error=0.20),        # noisy cooperator
}


def make_opponent(kind: str, seed: int = 0, **kwargs) -> StrategyAgent:
    """Strategy name -> StrategyAgent; presets can be overridden by
    kwargs."""
    cfg = dict(_PRESETS.get(kind, dict(kind=kind)))
    cfg.update(kwargs)
    return StrategyAgent(seed=seed, **cfg)


# ------------------------------------------------------ Trait switches (H1A)
#: The "capricious partner" scenarios of H1A; values are phase
#: cycles (120 rounds split into four 30-round phases).
SWITCH_SCENARIOS = {
    # reciprocity -> exploitation -> reconciliation -> reciprocity
    "recip_expl_recon": ("tft", "alld", "gtft", "tft"),
    # naive cooperation -> exploitation (a trust-baiting trap)
    "coop_trap": ("allc", "alld", "allc", "alld"),
    # outcome-conditional <-> exploitation (a WSLS-bearing cycle)
    "wsls_flip": ("wsls", "alld", "wsls", "gtft"),
}

#: Trait-switch period in rounds: 120 rounds / 4 phases = 30.
SWITCH_PERIOD = 30


def make_switching_opponent(scenario: str, n_rounds: int = 120,
                            seed: int = 0, error: float = 0.05
                            ) -> StrategyAgent:
    """
    Build a trait-switching opponent.

    Execution noise is supplied alongside, so "noise" and "intent
    change" coexist — the condition in which latent-trait inference
    actually pays.
    """
    cycle = SWITCH_SCENARIOS[scenario]
    sched = [(r, cycle[(r // SWITCH_PERIOD) % len(cycle)])
             for r in range(0, n_rounds, SWITCH_PERIOD)]
    return StrategyAgent(kind=cycle[0], seed=seed, error=error, schedule=sched)


def switch_rounds(n_rounds: int = 120) -> List[int]:
    """Rounds at which traits actually switch (excluding 0) — for
    aligned analyses."""
    return list(range(SWITCH_PERIOD, n_rounds, SWITCH_PERIOD))


# ------------------------------------------------ Recovery opponents (H6)
class LikelihoodAgent:
    """
    **An opponent that generates actions directly from
    OpponentInversion's generative likelihood.**

    Parameter-recovery (H6) only: actions are sampled with the true
    theta = (alpha, rho, omega, eta, beta, lambda_j) known, so
    estimates can be compared to ground truth — the standard recovery
    validity check that the model can invert its own assumed
    data-generating process.

        P(a_j = C) = sigmoid(beta*(alpha + rho*f + omega*g
                             + eta*f*g + s(lambda_j, p)))

    p (the belief about the focal cooperation rate) is a running mean
    of observed focal actions — the same construction as the
    inverter's `my_cooperation_rate`, so the likelihoods match
    exactly.
    """

    def __init__(self, alpha: float = 0.0, rho: float = 0.0,
                 omega: float = 0.0, eta: float = 0.0,
                 beta: float = 3.0, lambda_j: float = 0.5,
                 seed: int = 0):
        self.theta = dict(alpha=float(alpha), rho=float(rho),
                          omega=float(omega), eta=float(eta),
                          beta=float(beta), lambda_j=float(lambda_j))
        self.rng = np.random.default_rng(seed)
        self.my_last = COOP
        self.other_last = COOP
        self.other_actions: List[int] = []
        self.round = 0

    def _coop_prob(self) -> float:
        from AIF_IPD.core.constants import empathy_shift
        th = self.theta
        f = 1.0 - 2.0 * float(self.other_last)     # focal's last action
        g = 1.0 - 2.0 * float(self.my_last)        # own last action
        p = (float(np.mean([a == COOP for a in self.other_actions]))
             if self.other_actions else 0.5)
        z = th["beta"] * (th["alpha"] + th["rho"] * f + th["omega"] * g
                          + th["eta"] * f * g
                          + empathy_shift(th["lambda_j"], p))
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))

    def act(self) -> int:
        self.round += 1
        a = COOP if self.rng.random() < self._coop_prob() else DEFECT
        self.my_last = a
        return a

    def observe(self, focal_action: int) -> None:
        self.other_last = int(focal_action)
        self.other_actions.append(int(focal_action))

    def begin_partner(self, identity: int) -> None:
        return None


class ProbeAgent:
    """
    **An active stimulus policy for identifiability** (the H6
    focal).

    Recovery hinges on the design matrix (1, f, g, f*g) actually
    being spanned: an always-cooperating focal makes f constant,
    alpha and rho collinear, and rho/omega/eta unidentifiable.
    ProbeAgent cooperates i.i.d. with probability p_coop, balancing f
    across both levels and filling all four (f, g) cells.
    """

    def __init__(self, p_coop: float = 0.5, seed: int = 0):
        self.p_coop = float(p_coop)
        self.rng = np.random.default_rng(seed)

    def act(self) -> int:
        return COOP if self.rng.random() < self.p_coop else DEFECT

    def observe(self, focal_action: int) -> None:
        return None

    def begin_partner(self, identity: int) -> None:
        return None
