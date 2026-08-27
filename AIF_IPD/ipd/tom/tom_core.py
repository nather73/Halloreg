"""
ipd.tom.tom_core
================

Theory-of-Mind core: opponent prediction q(a_j | h_t) and the
**recursive social EFE**.

Albarracin et al. (2026) eq. (4):

    G_social(a_i) = (1-lam)*G_self(a_i)
                    + lam*E_{q(a_j)}[G_other(a_j)] + G_epistemic

Final form used here:

    G_social(a_i) = (1-lam)*(prag_self(a_i)  - IG_self(a_i))
                    +  lam *(prag_other(a_i) - IG_other(a_i))

  - prag_self  : cost form (-E[U]) of expected utility under my
                 payoffs; smaller is preferred.
  - prag_other : the same under the opponent's payoffs (perspective
                 taking — only the preference C is swapped).
  - IG_self    : my expected information gain about the opponent.
  - IG_other   : how much my action narrows **the opponent's belief
                 about me** (self-projection filter). Weighted by
                 lambda: the more empathic, the more value in actions
                 that help the opponent understand me.
  - The only weights are (1-lam) and lam — no separate epistemic
    weight, so lambda remains the sole convex coefficient between the
    self and other branches.

[Two recursions]
 (R1) depth-2 perspective taking: how the opponent sees "me" is
      computed by the self-projection filter and injected directly
      into the opponent model's believed_my_policy — recursion depth
      is built in without hand-mixed coefficients.
 (R2) recursive epistemic value: the opponent's information gain about
      me (IG_other) is included with lambda weighting; the original
      paper included only the opponent's pragmatic value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import COOP, DEFECT, PD_PAYOFFS, PAYOFF_SELF
from AIF_IPD.core.generative import efe_terms, softmax
from .inversion import OpponentInversion, ObservationContext


# --------------------------------------------------------------- static ToM
class TheoryOfMind:
    """
    Static ToM: the opponent as a rational agent structurally
    isomorphic to me.

        q(a_j) ~ exp(beta_j * negEFE_j(a_j))

    negEFE_j is the expected value under the opponent's payoffs,
    marginalised over the policy they believe I follow. It provides
    the **prior prediction** for the early rounds before the particle
    filter has seen data.
    """

    def __init__(self, beta_other: float = 4.0):
        self.beta_other = float(beta_other)
        # The (C, D) policy the opponent believes I follow.
        self._believed_my_policy = np.array([0.5, 0.5])

    def update_my_policy_belief(self, my_coop_rate: float) -> None:
        """Update the believed policy with my realised cooperation
        rate."""
        p = float(np.clip(my_coop_rate, 0.0, 1.0))
        self._believed_my_policy = np.array([p, 1.0 - p])

    def opponent_efe(self, believed_my_policy: Optional[np.ndarray] = None
                     ) -> np.ndarray:
        """negEFE of each opponent action under their payoffs,
        shape (2,) = [C, D]."""
        pi = (self._believed_my_policy if believed_my_policy is None
              else believed_my_policy)
        negG = np.zeros(2)
        for a_j in (COOP, DEFECT):
            val = 0.0
            for a_i in (COOP, DEFECT):
                _, other_payoff = PD_PAYOFFS[(a_i, a_j)]
                val += pi[a_i] * other_payoff
            negG[a_j] = val
        return negG

    def predict_opponent_action(self,
                                believed_my_policy: Optional[np.ndarray] = None
                                ) -> np.ndarray:
        """Static-ToM action distribution
        q(a_j) = softmax(beta_j * negEFE_j)."""
        return softmax(self.opponent_efe(believed_my_policy),
                       temperature=1.0 / self.beta_other)


# --------------------------------------------------------------- gated ToM
class GatedToM:
    """
    Reliability-gated ToM: smoothly interpolates the static prior and
    the learned posterior (particle filter) by the filter reliability r,

        q_gated = r * q_learned + (1 - r) * q_static,

    relying on structure when data are absent and on observed
    individual traits as they accumulate.
    """

    def __init__(self, tom: TheoryOfMind, inversion: OpponentInversion):
        self.tom = tom
        self.inversion = inversion

    def predict_opponent_action(self, ctx: Optional[ObservationContext]
                                ) -> np.ndarray:
        r = self.inversion.reliability()
        q = r * self.inversion.predict_action(ctx) \
            + (1.0 - r) * self.tom.predict_opponent_action()
        return q / q.sum()


# --------------------------------------------------------------- social EFE
@dataclass
class SocialEFEResult:
    """Result of one social-EFE computation."""
    G_social: np.ndarray          # (2,) social EFE per action (lower = preferred)
    q_response: np.ndarray        # (2,) opponent action prediction
    info: dict


class RecursiveSocialEFE:
    """
    Recursive social-EFE calculator.

    Parameters
    ----------
    gated_tom : GatedToM
    inversion : OpponentInversion            — opponent theta filter
    self_inversion : OpponentInversion|None  — self-projection filter
    empathy_factor : float                   — initial lambda (updated
                                               each round by the agent)
    beta_self : float                        — my action precision
    recursive_depth : int                    — 2 enables depth-2
    """

    def __init__(self, gated_tom: GatedToM, inversion: OpponentInversion,
                 empathy_factor: float = 0.4, beta_self: float = 4.0,
                 recursive_depth: int = 2,
                 self_inversion: Optional[OpponentInversion] = None):
        self.gated = gated_tom
        self.inversion = inversion
        self.self_inversion = self_inversion
        self.lam = float(empathy_factor)
        self.beta_self = float(beta_self)
        self.recursive_depth = int(recursive_depth)
        self.my_coop_rate = 0.5

    # ------------------------------------------------- Prediction (depth-2)
    def opponent_prediction(self, ctx: Optional[ObservationContext]
                            ) -> np.ndarray:
        """
        Opponent prediction q(a_j).

        At depth >= 2 the "policy the opponent believes I follow" is
        built from the self-projection filter's predicted cooperation
        and injected into the static ToM. From that filter's viewpoint
            f_me = the reciprocity stimulus the opponent saw from me
                   = their own last action,
            g_me = my own last action
        (f/g roles exactly swapped vs the focal filter).
        """
        r = self.inversion.reliability()
        q_learned = self.inversion.predict_action(ctx)

        if self.recursive_depth < 2 or self.self_inversion is None:
            q = r * q_learned + (1.0 - r) * self.gated.tom.predict_opponent_action()
            return q / q.sum()

        f_me = (0.0 if ctx is None or ctx.their_last_action is None
                else 1.0 - 2.0 * float(ctx.their_last_action))
        g_me = (0.0 if ctx is None or ctx.my_last_action is None
                else 1.0 - 2.0 * float(ctx.my_last_action))
        my_pc = self.self_inversion.predict_coop(f_me, g_me)
        believed_my = np.array([my_pc, 1.0 - my_pc])

        q_static_cond = self.gated.tom.predict_opponent_action(
            believed_my_policy=believed_my)
        q = r * q_learned + (1.0 - r) * q_static_cond
        return q / q.sum()

    # ------------------------------------------------- Step terms
    def step_terms(self, ctx: Optional[ObservationContext],
                   q: np.ndarray) -> dict:
        """
        Social-EFE components of my actions under prediction q,
        shared by the single-step path and any rollout step.
        """
        pc = float(q[COOP])

        # --- Pragmatic value, my viewpoint (cost form) ---
        prag = efe_terms(pc, PAYOFF_SELF)["pragmatic"]     # (2,) E[U]
        prag_self = -prag

        # --- Pragmatic value, their viewpoint (perspective) ---
        prag_other = np.zeros(2)
        for a_i in (COOP, DEFECT):
            val = 0.0
            for a_j in (COOP, DEFECT):
                _, other_payoff = PD_PAYOFFS[(a_i, a_j)]
                val += q[a_j] * other_payoff
            prag_other[a_i] = -val

        # --- Epistemic value: my gain about the opponent ---
        # Playing C makes next round's reciprocity signal f' = +1,
        # D makes it -1; the exact IG (v3.7.5) marginalises their
        # simultaneous move given the current context (f_now = my
        # last, g_now = their last).
        f_now = (0.0 if ctx is None or ctx.my_last_action is None
                 else 1.0 - 2.0 * float(ctx.my_last_action))
        g_now = (0.0 if ctx is None or ctx.their_last_action is None
                 else 1.0 - 2.0 * float(ctx.their_last_action))
        IG_self = np.array([
            self.inversion.expected_infogain_exact(+1.0, f_now, g_now),
            self.inversion.expected_infogain_exact(-1.0, f_now, g_now),
        ])

        # --- Epistemic value: their gain about me (lambda-weighted) ---
        IG_other = np.zeros(2)
        if self.self_inversion is not None:
            f_me = (0.0 if ctx is None or ctx.their_last_action is None
                    else 1.0 - 2.0 * float(ctx.their_last_action))
            g_me = (0.0 if ctx is None or ctx.my_last_action is None
                    else 1.0 - 2.0 * float(ctx.my_last_action))
            for a_i in (COOP, DEFECT):
                IG_other[a_i] = self.self_inversion.observed_infogain(
                    a_i, f_me, g_me)

        return {"prag_self": prag_self, "prag_other": prag_other,
                "IG_self": IG_self, "IG_other": IG_other,
                "pragmatic_utility": prag, "pc": pc}

    # ------------------------------------------------- Single step
    def compute(self, ctx: Optional[ObservationContext],
                lam: Optional[float] = None) -> SocialEFEResult:
        lam = self.lam if lam is None else float(lam)
        q = self.opponent_prediction(ctx)
        t = self.step_terms(ctx, q)

        self_branch = t["prag_self"] - t["IG_self"]
        other_branch = t["prag_other"] - t["IG_other"]
        G_social = (1.0 - lam) * self_branch + lam * other_branch

        return SocialEFEResult(
            G_social=G_social, q_response=q,
            info={"pc": t["pc"], "lam": lam,
                  "reliability": self.inversion.reliability(),
                  # Expected payoff of the chosen action —
                  # diagnostic only (RPE is computed separately by
                  # CoreAffect from the baseline reward distribution).
                  "pragmatic_utility": t["pragmatic_utility"],
                  "IG_self": t["IG_self"], "IG_other": t["IG_other"]})

    def select_action(self, ctx: Optional[ObservationContext],
                      lam: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[int, SocialEFEResult]:
        """Sample an action from softmax(-G_social)."""
        res = self.compute(ctx, lam)
        q_pi = softmax(-res.G_social, temperature=1.0 / self.beta_self)
        rng = rng or np.random.default_rng()
        action = COOP if rng.random() < q_pi[COOP] else DEFECT
        res.info["q_pi"] = q_pi
        res.info["action"] = action
        return action, res
