"""
core.empathy
============

**Empathy — the legacy integrator of the empathy weight lambda.**

(Retained for the EmpathicAgent baseline; HalloRegAgent replaced this
integrator with the allostatic direct mapping in agent._regulate and
keeps this object only as the lambda carrier.)

lambda weights the opponent's expected free energy in action
selection (Albarracin et al. 2026, eq. 4), an **exogenous constant**
in the original paper. This module derives it endogenously from two
drive signals:

    lambda_aff (affective motivation) — from CoreAffect,
        valence * arousal: "which way does unpredicted loss/gain push
        me"; the interoceptive route.
    lambda_ctx (contextual motivation) — from OpponentInversion,
        the standardised sum of the inferred alpha and lambda_j:
        "what kind of person have I inferred this partner to be"; the
        exteroceptive route.

    lambda_t = lambda_{t-1}
               + eta * [(1 - w_cd)*lambda_aff + w_cd*lambda_ctx]

lambda is thus an **integrator** whose state carries the accumulated
relationship history; lambda_{t=0} comes from the SelfModel setpoint.
The gain eta sets the integration timescale (a neuromodulatory gain;
eta = 0.05 gives a ~20-round time constant). lambda_ctx standardises
its two inputs onto a signed [-1, +1] prosociality scale,

    lambda_ctx = 1/2 * [tanh(alpha/a_scale) + (2*lambda_j - 1)],

a monotone transform preserving "larger alpha and lambda_j -> larger
lambda_ctx" while removing the scale mismatch and the 0.5 neutrality
offset.

Channel asymmetry: lambda_aff is an **error** signal
(self-extinguishing once expectations adapt); lambda_ctx is a
**level** signal (a confidently inferred exploiter keeps pulling
lambda to its floor) — a dual-timescale design of surprise-driven
immediacy plus inference-driven persistence.
"""

from __future__ import annotations

import numpy as np


class Empathy:
    """
    The lambda integrator (legacy baseline machinery).

    Parameters
    ----------
    lam_init : float
        lambda_{t=0}, supplied by SelfModel.lambda_setpoint().
    w_cd : float in [0, 1]
        Affective vs contextual channel weight (0 = purely affective,
        1 = purely inferential).
    gain : float
        Integrator gain eta.
    lam_min, lam_max : float
        Admissible lambda interval; as a convex EFE weight lambda must
        stay within [0, 1].
    alpha_scale : float
        a_scale of the lambda_ctx standardisation.
    """

    def __init__(self, lam_init: float, w_cd: float = 0.5,
                 gain: float = 0.05, w_tonic: float = 0.10,
                 w_sp: float = 1.0, aff_gain: float = 0.30, lam_min: float = 0.0,
                 lam_max: float = 0.80, alpha_scale: float = 2.0,
                 gain_down: float = 0.45):
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        #: Downward relaxation rate (threat learning); upward uses
        #: self.gain.
        self.gain_down = float(gain_down)
        self.lam_init = float(np.clip(lam_init, lam_min, lam_max))
        self.lam = self.lam_init
        self.w_cd = float(np.clip(w_cd, 0.0, 1.0))
        self.gain = float(gain)
        self.w_tonic = float(w_tonic)   # retired v1.8.0 (signature compat)
        self.w_sp = float(w_sp)         # setpoint relaxation strength
        self.aff_gain = float(aff_gain) # affect gain (round-level jumps)
        self.alpha_scale = float(alpha_scale)

        # Latest step diagnostics
        self.last = {"lambda": self.lam, "lambda_aff": 0.0,
                     "lambda_ctx": 0.0, "drive": 0.0}

    # ------------------------------------------------------------ lambda_ctx
    def contextual(self, alpha_hat: float, lambda_j_hat: float) -> float:
        """
        Inferred partner traits -> contextual drive
        lambda_ctx in [-1, +1] (the standardised sum above).
        """
        a = float(np.tanh(float(alpha_hat) / max(self.alpha_scale, 1e-6)))
        l = 2.0 * float(np.clip(lambda_j_hat, 0.0, 1.0)) - 1.0
        return 0.5 * (a + l)

    # ------------------------------------------------------------ One step
    def step(self, lambda_aff: float, lambda_ctx: float = 0.0,
             valence: float = 0.0, lam_sp: Optional[float] = None,
             threat: bool = False) -> float:
        """
        **Two separated timescales** (v1.8.0).

            lambda_t = clip(lambda_{t-1}
                            + eta_sp*(lam_sp - lambda_{t-1})   (slow)
                            + g_aff*lambda_aff)                (fast)

          - lam_sp (setpoint) — **between-relationship** fitness ("is
            this partner good relative to others I know"), moving
            gradually (eta_sp = gain).
          - lambda_aff = V*A — the phasic **between-state** response
            ("is this situation good within this relationship"), which
            must be able to jump within a round (g_aff = aff_gain).

        A single shared gain capped lambda at 0.05/round and could not
        follow one-round state switches (e.g. escaping WSLS's CD trap
        needs a one-round defection detour); the setpoint is a
        property of the relationship (slow), affect a property of the
        situation (fast) — binding them to one gain was the design
        error.
        """
        drive_sp = 0.0
        # **Asymmetric relaxation** (v2.8.0) — threat fast, trust
        # slow. With eta = 0.05 the descent to lambda = 0 against an
        # exploiter took ~100 rounds (measured), so eta_down = 0.45
        # gives a ~2-round time constant. The asymmetry is principled:
        # betrayal learning is fast and trust recovery slow, and
        # allostatically the threat-direction predictive response
        # should engage first. Fast relaxation applies **only under
        # threat** (relationship fitness Phi < 0): keying on
        # lam_sp < lambda alone also fires in good relationships
        # (measured pathology vs ALLC: lambda 0.386 -> 0.282) — the
        # asymmetry is about *bad relationships*, not about the
        # downward direction as such.
        eta = self.gain
        if lam_sp is not None:
            drive_sp = self.w_sp * (float(lam_sp) - self.lam)
            if threat and float(lam_sp) < self.lam:
                eta = self.gain_down
        self.lam = float(np.clip(
            self.lam + eta * drive_sp + self.aff_gain * float(lambda_aff),
            self.lam_min, self.lam_max))
        self.last = {"lambda": self.lam, "lambda_aff": float(lambda_aff),
                     "lambda_ctx": 0.0, "valence": float(valence),
                     "lam_sp": float(lam_sp) if lam_sp is not None
                     else float("nan"),
                     "drive": self.gain * drive_sp
                     + self.aff_gain * float(lambda_aff)}
        return self.lam

    def reset(self, lam_init: float = None) -> float:
        """
        Reset lambda at the start of a new relationship — to the
        supplied value (the updated SelfModel setpoint) or, if absent,
        to the original one.
        """
        if lam_init is not None:
            self.lam_init = float(np.clip(lam_init, self.lam_min, self.lam_max))
        self.lam = self.lam_init
        return self.lam


def empathy_shift_z(lam, adv_self, adv_other):
    """
    **Z-based empathy intercept** es(lambda, s), v2.1:

        es(lambda, s) = (1-lambda)*[Z_self(s,C) - Z_self(s,D)]
                      +   lambda  *[Z_other(s,C) - Z_other(s,D)]

    Why (lambda, s) rather than (lambda, p): the analytic
    empathy_shift(lambda, p) = (T-S)lambda + (R-T+P-S)p + (S-P) was a
    1-step solution that (i) presupposes payoff knowledge
    (contradicting payoff_access="naive"), (ii) cannot see that
    defecting now worsens the next K rounds, and (iii) is
    state-independent, unable to distinguish CD from DD. The Z form
    resolves all three — Z is learned from observation, carries the
    gamma-horizon future, and is indexed by (state, action).

    Known limitation: Z is an **on-policy** value, so an already
    cooperative policy self-reinforces through A_self, and the measure
    is "the value of what I am doing", not "the best possible".

    Indifference point: es = 0 iff
    lambda* = A_self / (A_self - A_other).
    """
    return (1.0 - lam) * adv_self + lam * adv_other

