"""
core.core_affect
================

**CoreAffect — two-dimensional core affect: valence x arousal.**

Premise: primary reward acquisition serves homeostasis over the
internal model, and **the expected-reward distribution stands in as
the generative model of interoception**. CoreAffect reduces the
relation between that model and observation to two low-dimensional
affect axes.

valence — fitness within the social population (a LEVEL signal)
----------------------------------------------------------------
    valence = 2 * F-hat_ref(median(Q_current_partner)) - 1

Q_current_partner : the expected-reward (value) distribution of the
current partner — an EMA of the QRTD Z(s,a) over experienced
situations (SelfModel.update_partner_value).
F-hat_ref : the CDF within the distance-weighted reference (a
Wasserstein barycentre) of **other partners'** distributions.

Valence is **not a prediction error**: it is the subjective fitness of
the current generative model within SelfModel's beliefs about the
social environment. Facing ALLD, the learned expectation for this
partner falls below those of remembered others — that is negative
valence. Because the reference is other people, not one's own past,
valence does not extinguish under chronic exploitation (with temporal
self-comparison the reference catches up and valence was measured to
vanish or invert). The current partner is excluded from the
population.

arousal — the update magnitude of the generative model (an ERROR
signal)
----------------------------------------------------------------
    arousal = 1 - exp(-W1(Z_{t-1}, Z_t) / span_V / kappa)

The mismatch between successive expected-reward distributions — the
model-update magnitude. What surprises is not the immediate reward
but the value of the situation: one betrayal after long cooperation
barely moves the reward expectation yet collapses the prospect (Z).

Division of labour: valence is the **level** (persistent — quality of
the relationship), arousal is the **error** (self-extinguishing —
the predicted is not surprising). Since lambda_aff = valence *
arousal, only unpredicted events move lambda, and in the direction of
the relationship's quality.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .self_model import SelfModel

_EPS = 1e-12


class CoreAffect:
    """
    Core-affect generator (v1.5.0 — QRTD-grounded).

    Parameters
    ----------
    self_model : SelfModel
        Memory store; updates the current partner's value distribution
        and supplies the population reference.
    payoffs : (4,) array
        Current payoff vector (normalisation constants).
    kl_scale : float
        Arousal saturation constant kappa.
    """

    def __init__(self, self_model: SelfModel, payoffs: np.ndarray,
                 kl_scale: float = 0.05, sigma_floor: float = 0.5,
                 obs_weight: float = 1.0, arousal_floor: float = 0.05):
        self.self_model = self_model
        self.payoffs = np.asarray(payoffs, dtype=float).copy()
        self.kl_scale = float(kl_scale)
        self.arousal_floor = float(arousal_floor)
        self.sp_disposition = 0.50   # m — magnitude of the dispositional part
        self.sp_i0 = 0.0             # I0 — target intercept
        self.sigma_floor = float(sigma_floor)   # signature compat (unused)
        self.obs_weight = float(obs_weight)     # signature compat (unused)
        self.self_model.set_payoff_scale(self.payoffs)
        self.identity: Optional[int] = None
        self.last = {"valence": 0.0, "arousal": 0.0, "lambda_aff": 0.0,
                     "rpe": 0.0, "surprise": 0.0, "value": float("nan")}

    def set_payoffs(self, payoffs: np.ndarray) -> None:
        """Non-stationary payoff alignment — refresh normalisation."""
        self.payoffs = np.asarray(payoffs, dtype=float).copy()
        self.self_model.set_payoff_scale(self.payoffs)

    def begin_partner(self, identity: Optional[int]) -> None:
        self.identity = identity

    # ------------------------------------------------------------ One step
    def step(self, observed_state: int,
             opponent_cooperated: Optional[bool] = None,
             value_vector: Optional[np.ndarray] = None,
             value_shift: Optional[float] = None,
             z_snapshot: Optional[np.ndarray] = None,
             state_valence: Optional[float] = None,
             adv: Optional[tuple] = None) -> dict:
        """
        Construct affect from the observed joint outcome of round t-1.

        value_vector : Z quantile vector of the just-experienced
                       (state, action) — EMA material for the current
                       partner's expected-reward distribution.
        value_shift  : W1(Z_before, Z_after)/span_V — model-update
                       magnitude (arousal).
        z_snapshot   : full Z_self snapshot, stored for re-encounter
                       restoration.

        Steps: (1) value-distribution EMA -> (2) valence ->
        (3) arousal -> (4) lambda_aff = V*A -> (5) commit cooperation.
        """
        s = int(observed_state)
        r_obs = float(self.payoffs[s])

        # --- 1. Update the partner's expected-reward distribution ---
        if value_vector is not None:
            self.self_model.update_partner_value(
                self.identity, value_vector, z_snapshot=z_snapshot)

        # --- 2. valence: **between-state** fitness (is the current
        #     state good within this relationship):
        #     valence = 2 * F-hat_{s'}(V(s_now)) - 1, a d(s')-weighted
        #     percentile of state values. Between-relationship
        #     comparison lives separately in the lambda setpoint.
        valence = float(state_valence) if state_valence is not None else 0.0

        # --- 2b. Lambda setpoint: **between-relationship** fitness ---
        if adv is not None:
            # v2.0 — compensatory: baseline at the empathy this
            # relationship requires.
            lam_sp, fitness, _base = self.self_model.lambda_sp_compensatory(
                self.identity, adv[0], adv[1], m=self.sp_disposition,
                i0=self.sp_i0)
        else:
            fitness = self.self_model.social_fitness(self.identity)
            lam_sp = float(np.clip(
                0.5 * (1.0 + fitness), self.self_model.lam_floor,
                self.self_model.lam_ceil))

        ent = self.self_model.memory.get(self.identity)
        my_med = (float(ent.value_dist[len(ent.value_dist) // 2])
                  if ent is not None and ent.value_dist is not None
                  else float("nan"))
        rpe = (my_med - self.self_model.reference_median(exclude=self.identity)
               if np.isfinite(my_med) else 0.0)

        # --- 3. arousal: model-update magnitude (error) ---
        # **Floor a_min > 0.** If arousal converges to 0,
        # lambda_aff = V*A vanishes and affect loses all leverage on
        # lambda (especially in fully learned relationships); the
        # floor keeps a minimal channel open for the level signal.
        surprise = float(value_shift) if value_shift is not None else 0.0
        arousal = float(1.0 - np.exp(-surprise / max(self.kl_scale, _EPS)))
        arousal = float(max(arousal, self.arousal_floor))

        # --- 4. Affective drive ---
        lambda_aff = valence * arousal

        # --- 5. Memory commit (cooperation rate -> setpoint) ---
        self.self_model.commit_observation(
            self.identity, opponent_cooperated=opponent_cooperated)

        self.last = {
            "valence": float(valence), "arousal": arousal,
            "lambda_aff": float(lambda_aff), "rpe": float(rpe),
            "surprise": surprise, "tau_hat": float((valence + 1.0) / 2.0),
            "r_base": self.self_model.reference_median(exclude=self.identity),
            "r_mean": self.self_model.reference_median(exclude=self.identity),
            "r_obs": r_obs, "value": my_med,
            "fitness": float(fitness), "lam_sp": lam_sp,
            "pessimism": 0.0,
        }
        return dict(self.last)

    # ------------------------------------------------------------ Diagnostics
    def expected_reward(self) -> float:
        ent = self.self_model.memory.get(self.identity)
        if ent is None or ent.value_dist is None:
            return 0.0
        return float(np.mean(ent.value_dist))

    def pessimism(self) -> float:
        return 0.0
