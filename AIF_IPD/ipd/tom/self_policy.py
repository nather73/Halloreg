"""
ipd.tom.self_policy
===================

**SelfPolicy — policy selection in trait space.**

(Legacy trait path: this is the action-selection machinery of the
EmpathicAgent baseline; HalloRegAgent uses the lambda-only path and
keeps this component as a unit-tested instrument.)

What the focal agent chooses each round is not an action a_i in
{C, D} but its **own traits** theta_i = (rho, omega, eta); the action
is then sampled from the trait likelihood

    P(a_i = C | h, theta_i, lam)
        = sigmoid(beta * (rho*f + omega*g + eta*f*g + s(lam, p_j)))

  - alpha_i is **pinned to 0**: unconditional cooperation bias is
    excluded by construction, so all cooperation must be reciprocal or
    empathic. The intercept does not vanish — s(lam, p) moves with
    lambda, so **lambda plays the role of a dynamic intercept**.
  - beta_i is fixed at 4.0: decision precision is not part of the
    prosociality construct and is kept out of regulation.

Why trait space rather than action space: Albarracin et al. (2026)
introduced theta to compress the horizon-exploding latent state via
Theory of Mind, and that compression must apply to the self policy as
well — otherwise self/other representations are asymmetric and
perspective taking is ill-defined. Computationally, action-level
planning branches 2^H, while in trait space the (f, g) joint has only
4 states and distributions can be forward-propagated at O(4H).

Trait EFE:
    G(theta_i | theta_j, lam, ctx)
        = -[E[u_lam] + w_epi*(IG_j + IG_R)] + w_cplx*KL(theta_i||prior)
The likelihood's s(lam, p) is the exact 1-step analytic solution of
the lambda-weighted pragmatic value, so no separate
(1-lam)self + lam*other composition is repeated inside the EFE. The
epistemic terms depend on theta_i because theta_i changes the (f, g)
state distribution and hence which observations will be realised —
**endogenous probing** emerges from the objective.

Candidate generation is sequential Monte Carlo, exactly symmetric to
OpponentInversion (propose -> evaluate -> weight by exp(-gamma G),
i.e. the standard policy posterior q(pi) ~ exp(-G(pi)) -> resample),
so "posterior becomes next round's prior" holds automatically.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import (
    COOP, DEFECT, PAYOFF_OTHER, PAYOFF_SELF, empathy_shift, joint_index,
)

_EPS = 1e-12

#: Self-trait axes; alpha and beta are fixed, hence excluded.
SELF_AXES = ("rho", "omega", "eta")

#: Priors per axis (mean, sd), starting neutral (0): seeding an
#: initial reciprocity would make "reciprocity emerges from inference"
#: an artefact of initialisation.
SELF_PRIOR = {"rho": (0.0, 1.0), "omega": (0.0, 0.6), "eta": (0.0, 0.6)}

#: Trait bounds; with beta = 4 the sigmoid saturates by |rho| ~ 3.
SELF_BOUNDS = {"rho": (-3.0, 3.0), "omega": (-2.0, 2.0), "eta": (-2.0, 2.0)}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class SelfPolicy:
    """
    Sequential Monte Carlo policy over self traits (rho, omega, eta).

    Parameters
    ----------
    n_particles : int
        Particle count K.
    prop_sd : float
        Proposal diffusion sd; large = unstable stance, small = slow
        adaptation.
    gamma : float
        Policy precision; larger sharpens EFE differences.
    horizon : int
        Trait-space rollout horizon H.
    discount : float
        Temporal discount.
    w_epi_j, w_epi_r : float
        Epistemic weights (0 = purely pragmatic ablation).
    beta_self : float
        Fixed likelihood precision beta.
    resample_frac : float
        Resample when ESS falls below this fraction of K.
    """

    def __init__(self, n_particles: int = 64, prop_sd: float = 0.40,
                 gamma: float = 8.0, horizon: int = 6, discount: float = 0.9,
                 w_epi_j: float = 10.0, w_epi_r: float = 1.0,
                 w_cplx: float = 0.15,
                 beta_self: float = 4.0, resample_frac: float = 0.5,
                 seed: Optional[int] = None):
        self.K = int(n_particles)
        self.prop_sd = float(prop_sd)
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.discount = float(discount)
        # The two epistemic weights are **separate** (v1.6.1): IG_j
        # effectively vanishes within ~20 rounds as the opponent
        # posterior converges, while IG_R persists — their magnitudes
        # diverge by orders of magnitude, so one shared weight cannot
        # serve both.
        self.w_epi_j = float(w_epi_j)
        self.w_epi_r = float(w_epi_r)
        self.w_cplx = float(w_cplx)
        self.beta = float(beta_self)
        self.resample_frac = float(resample_frac)
        self.rng = np.random.default_rng(seed)

        self.theta: Dict[str, np.ndarray] = {}
        self._init_particles()
        self.weights = np.ones(self.K) / self.K
        self.last_G = np.zeros(self.K)

    # ============================================================ Init
    def _init_particles(self) -> None:
        for ax in SELF_AXES:
            mu, sd = SELF_PRIOR[ax]
            lo, hi = SELF_BOUNDS[ax]
            self.theta[ax] = np.clip(
                self.rng.normal(mu, sd, self.K), lo, hi)

    def set_prior(self, prior: Optional[Dict[str, tuple]]) -> None:
        """
        Re-initialise particles from the SelfModel-supplied prior on
        re-encounter: "this is the stance I used with this person".
        """
        if not prior:
            return
        for ax in SELF_AXES:
            if ax not in prior:
                continue
            mu, sd = prior[ax]
            lo, hi = SELF_BOUNDS[ax]
            self.theta[ax] = np.clip(
                self.rng.normal(float(mu), max(float(sd), 1e-3), self.K),
                lo, hi)
        self.weights = np.ones(self.K) / self.K

    # ============================================================ Likelihood
    def _coop_prob(self, th: Dict[str, np.ndarray], f: float, g: float,
                   lam: float, p_other: float,
                   shift: Optional[float] = None) -> np.ndarray:
        """
        Own-action likelihood P(a_i = C | f, g, theta_i, lam).

        **Basis role convention (swapped vs the opponent model):**
          - f = the opponent's last action -> what I reciprocate
          - g = my own last action -> self-inertia
        In the opponent likelihood f was "my last action"; missing
        this swap silently inverts reciprocity. p_other is the
        **opponent's** cooperation rate here.
        """
        # Use the supplied shift when given (naive mode's learned
        # s-hat); otherwise assume payoff knowledge and use the
        # analytic solution (oracle mode).
        if shift is None:
            shift = empathy_shift(lam, p_other)
        z = self.beta * (th["rho"] * f + th["omega"] * g
                         + th["eta"] * f * g + shift)
        return _sigmoid(z)

    def coop_prob_mixture(self, f: float, g: float, lam: float,
                          p_other: float,
                          shift: Optional[float] = None) -> float:
        """Particle-weighted mean cooperation probability, used for
        actual action sampling."""
        pc = self._coop_prob(self.theta, f, g, lam, p_other, shift)
        return float(np.sum(self.weights * pc))

    # ============================================================ Trait EFE
    def _evaluate(self, th: Dict[str, np.ndarray], f: float, g: float,
                  lam: float, p_other: float, p_coop_j: float,
                  shift_i: Optional[float],
                  u_self: Optional[np.ndarray],
                  u_other: Optional[np.ndarray],
                  terminal, ig_j: Optional[np.ndarray],
                  ig_r: Optional[np.ndarray]) -> np.ndarray:
        """
        **Trait EFE — 1-step pragmatic + terminal Z + two epistemic
        terms.**

            G(theta_i) = -[E[u_lam] + w_epi*(IG_j + IG_R)]
                         + w_cplx * KL(theta_i || prior)

        [Rollout retired — v1.6.0] The bootstrapped return Z already
        contains the future, so re-accumulating it over a rollout
        would count the same term H+1 times. Evaluation is therefore
        1-step: marginalise each action's immediate utility over the
        opponent's action probability, and let the terminal Z summarise
        the rest.

        [Self/other symmetry] Other-utility uses the learned
        R-hat_other; the opponent's action is marginalised with
        p_coop_j from OpponentInversion; the terminal value includes
        Z_other, so tenses are symmetric.

        [Two epistemic terms] ig_j — information gain about the
        opponent's hidden intent; ig_r — information gain about the
        latent environment structure R-hat. Unexperienced outcomes
        (e.g. DC in a settled cooperative relationship) keep wide
        posteriors and earn high IG; once visited, IG vanishes.
        Exploration comes from the objective, not from optimistic
        initialisation — the unknown is "unknown", not "good".
        """
        US = PAYOFF_SELF if u_self is None else np.asarray(u_self, float)
        UO = PAYOFF_OTHER if u_other is None else np.asarray(u_other, float)
        pc_i = self._coop_prob(th, f, g, lam, p_other, shift_i)  # (K,)
        pj = float(np.clip(p_coop_j, 0.0, 1.0))

        total = np.zeros(self.K)
        # Current state index (f = their last, g = my last)
        o_prev = int(round((1.0 - f) / 2.0)) if f != 0.0 else 0
        m_prev = int(round((1.0 - g) / 2.0)) if g != 0.0 else 0
        s_cur = 2 * m_prev + o_prev

        for a_i in (COOP, DEFECT):
            p_i = pc_i if a_i == COOP else (1.0 - pc_i)
            # --- 1-step pragmatic, marginalised over their action ---
            u = 0.0
            for a_j, p_j in ((COOP, pj), (DEFECT, 1.0 - pj)):
                idx = joint_index(a_i, a_j)
                u += p_j * ((1.0 - lam) * US[idx] + lam * UO[idx])
            # --- Terminal value: Z summarises the future ---
            if terminal is not None:
                s_next = 2 * a_i + (COOP if pj >= 0.5 else DEFECT)
                u += self.discount * terminal(s_next, pj, lam)
            # --- Two epistemic strands ---
            epi = 0.0
            if ig_j is not None:
                epi += self.w_epi_j * float(ig_j[a_i])
            if ig_r is not None:
                epi += self.w_epi_r * float(ig_r[a_i])
            total = total + p_i * (u + epi)
        return total

    def step(self, theta_j: Dict[str, float], lam: float,
             f: float, g: float, p_other: float, p_self: float,
             ig_j: Optional[np.ndarray] = None,
             u_self: Optional[np.ndarray] = None,
             u_other: Optional[np.ndarray] = None,
             terminal=None, shift_i: Optional[float] = None,
             p_coop_j: float = 0.5,
             ig_r: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        One SMC round: propose -> evaluate -> weight -> resample.
        Returns posterior-mean traits and diagnostics.
        """
        # --- 1. Propose (diffusion kernel) ---
        for ax in SELF_AXES:
            lo, hi = SELF_BOUNDS[ax]
            self.theta[ax] = np.clip(
                self.theta[ax] + self.rng.normal(0.0, self.prop_sd, self.K),
                lo, hi)

        # --- 2. Evaluate ---
        value = self._evaluate(self.theta, f, g, lam, p_other, p_coop_j,
                               shift_i, u_self, u_other, terminal,
                               ig_j, ig_r)
        G = -value + self.w_cplx * self._complexity()
        self.last_G = G

        # --- 3. Weight: q(pi) ~ exp(-gamma G) ---
        logw = -self.gamma * G
        logw -= logw.max()
        w = np.exp(logw)
        s = float(w.sum())
        self.weights = (w / s) if s > _EPS else np.ones(self.K) / self.K

        # --- 4. Resample ---
        ess = 1.0 / float(np.sum(self.weights ** 2))
        if ess < self.resample_frac * self.K:
            self._resample()

        out = {ax: float(np.sum(self.weights * self.theta[ax]))
               for ax in SELF_AXES}
        out["ess"] = ess
        out["G_mean"] = float(np.sum(self.weights * G))
        return out

    def _complexity(self) -> np.ndarray:
        """
        **Complexity term** — KL cost from the prior (Mahalanobis^2/2
        under a Gaussian prior):

            C(theta_i) = sum_ax (theta_ax - mu_ax)^2 / (2 sigma_ax^2)

        A standard component of variational free energy, and essential
        here: without it traits drift without evidence. Concretely,
        against a constant opponent (ALLD) f is pinned at -1, so
        rho*f = -rho becomes a de-facto intercept — rho and the
        intercept turn collinear and rho gets selected as a
        **defection device** rather than reciprocity (measured
        dissociation -0.22 without this term). With it, reciprocity
        survives only when it actually creates value, i.e. when the
        opponent responds — the term enforces the construct
        "reciprocity is meaningful only toward responsive partners".
        """
        c = np.zeros(self.K)
        for ax in SELF_AXES:
            mu, sd = SELF_PRIOR[ax]
            c = c + (self.theta[ax] - mu) ** 2 / (2.0 * max(sd, 1e-6) ** 2)
        return c

    def _resample(self) -> None:
        """Systematic resampling; weights return to uniform."""
        pos = (self.rng.random() + np.arange(self.K)) / self.K
        idx = np.searchsorted(np.cumsum(self.weights), pos)
        idx = np.clip(idx, 0, self.K - 1)
        for ax in SELF_AXES:
            self.theta[ax] = self.theta[ax][idx]
        self.weights = np.ones(self.K) / self.K

    # ============================================================ Diagnostics
    def posterior_means(self) -> Dict[str, float]:
        return {ax: float(np.sum(self.weights * self.theta[ax]))
                for ax in SELF_AXES}

    def posterior_stds(self) -> Dict[str, float]:
        out = {}
        for ax in SELF_AXES:
            mu = float(np.sum(self.weights * self.theta[ax]))
            var = float(np.sum(self.weights * (self.theta[ax] - mu) ** 2))
            out[ax] = float(np.sqrt(max(var, 0.0)))
        return out
