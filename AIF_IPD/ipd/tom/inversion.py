"""
ipd.tom.inversion
=================

**OpponentInversion — particle-filter Bayesian inference over the
opponent's traits theta_j.**

This module implements "(i) perspective-taking, approximated via
particle filter Bayesian inference over the opponent's model
parameters".

[Generative likelihood — (1, f, g, f*g) basis]
Each particle carries one behavioural-profile hypothesis theta_k.

    P(a_j = C | h_t, θ_k)
        = σ( β_k · ( α_k + ρ_k·f + ω_k·g + η_k·f·g + s(λ_k, p) ) )

    f      : reciprocity signal of the focal agent's last action.
             cooperate -> +1, defect -> -1, no history -> 0
    g      : the opponent's **own** last-action signal (+1 / -1)
    alpha  : cooperation bias (high -> ALLC-like, low -> ALLD-like)
    rho    : reciprocity (positive -> TFT-like; reacts to my last move)
    omega  : inertia / self-consistency (repeats own last action)
    eta    : outcome-conditionality (f*g interaction — WSLS signature)
    beta   : behavioural precision (high = deterministic/intentional)
    lambda_j : the opponent's empathy weight (recursive latent)
    es     : the empathy shift toward cooperation (mirror es_z by
             default; analytic empathy_shift as fallback)

[Why the (1, f, g, f*g) basis is necessary — H1 identifiability]
A memory-one strategy is fully determined by cooperation probabilities
over the 4 combinations of (own last, other's last). Spanning that
space needs 4 degrees of freedom, which (1, f, g, f*g) provides.
In particular WSLS ("stay after a win, shift after a loss") reduces to
next_sign = f * g — a **pure eta axis**. With only (1, f), WSLS is
inexpressible and necessarily misclassified.

[Tense convention — guarding against off-by-one]
In a simultaneous game the action observed at round k is opp_{k-1},
which reacted to what the opponent had seen **before that**:

    path      | target          | f (my last)      | g (their last)
    ----------|-----------------|------------------|------------------
    update    | obs   opp_{k-1} | my_{k-2}         | opp_{k-2}
    decision  | pred  opp_k     | my_{k-1} (fresh) | opp_{k-1} (fresh)

Both update-path features point one step further into the past. The
caller (agent.py) is responsible for passing the correct tense in
ObservationContext.

[SelfModel link]
`set_prior(prior)` injects the (theta, dist) stored by SelfModel and
resamples the particles: uninformative for a new partner, resumed from
the remembered relationship on re-encounter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import COOP, empathy_shift

_EPS = 1e-10

# Axis order — must match core.self_model.THETA_AXES exactly.
THETA_AXES = ("alpha", "rho", "omega", "eta", "beta", "lambda_j")

# Physical bounds per axis (particle clipping).
#   beta > 0        : precision is positive.
#   lambda_j in [0,1]: convex-combination weight.
#   rho/omega/eta are logit coefficients — clipped generously only to
#   prevent numerical blow-up.
_BOUNDS = {
    "alpha": (-6.0, 6.0),
    "rho": (-4.0, 4.0),
    "omega": (-4.0, 4.0),
    "eta": (-4.0, 4.0),
    "beta": (0.05, 12.0),
    "lambda_j": (0.0, 1.0),
}


def _logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


@dataclass
class ObservationContext:
    """Round context for the likelihood (tense is the caller's
    responsibility)."""
    my_last_action: Optional[int] = None      # source of f
    their_last_action: Optional[int] = None   # source of g
    joint_outcome: Optional[int] = None
    round_number: int = 0


class OpponentInversion:
    """
    Sequential importance resampling (SIR) particle filter over the
    opponent's theta_j = (alpha, rho, omega, eta, beta, lambda_j).

    Parameters
    ----------
    n_particles : int
        Particle count N; >= 400 recommended for the 6-D posterior.
    resample_frac : float
        Resample when ESS drops below resample_frac * N.
    jitter_scale : float
        Roughening scale added after resampling. It prevents particle
        degeneracy and implements the assumption that **theta may
        drift slowly over time** — the reason trait-switch tracking
        (H1A) works at all. 0 collapses the filter to a point mass.
    """

    # Uninformative prior per axis (mean, sd)
    _PRIOR = {
        "alpha": (0.0, 2.0),
        "rho": (0.5, 1.2),
        "omega": (0.0, 1.0),
        "eta": (0.0, 1.0),
        "beta": (3.0, 1.8),
        # The lambda_j prior mean is **0.5 (neutral)** so that a
        # zero-observation posterior carries no artefactual drift.
        "lambda_j": (0.5, 0.3),
    }
    # Default resampling jitter sd per axis
    _JITTER = {"alpha": 0.10, "rho": 0.10, "omega": 0.10,
               "eta": 0.10, "beta": 0.15, "lambda_j": 0.05}

    def __init__(self, n_particles: int = 400, resample_frac: float = 0.5,
                 jitter_scale: float = 1.0, seed: int = 0):
        self.N = int(n_particles)
        self.resample_frac = float(resample_frac)
        self.jitter_scale = float(jitter_scale)
        self.rng = np.random.default_rng(seed)

        # The opponent's belief about my cooperation rate p — the
        # argument of the analytic empathy_shift fallback.
        self.my_cooperation_rate = 0.5
        #: es provider (v3.7.1): callable(lam_vec, f, g) -> shift
        #: vector. None -> analytic empathy_shift(lambda_j, p). The
        #: default agent injects the **mirror es_z** provider so the
        #: ToM assumes the same utility family that actually generates
        #: the focal agent's behaviour.
        self.es_provider = None

        # Current prior (replaced on SelfModel injection)
        self._prior = {ax: self._PRIOR[ax] for ax in THETA_AXES}
        self._init_particles()

    # ============================================================ Init
    def _init_particles(self) -> None:
        """Sample particles from the current prior with uniform
        weights."""
        self.theta = {}
        for ax in THETA_AXES:
            mu, sd = self._prior[ax]
            lo, hi = _BOUNDS[ax]
            self.theta[ax] = np.clip(
                self.rng.normal(mu, max(sd, 1e-6), self.N), lo, hi)
        self.weights = np.ones(self.N) / self.N
        self._h0_cache = None

    def set_prior(self, prior: Dict[str, tuple], reinit: bool = True) -> None:
        """
        Inject the theta prior {axis: (mean, sd)} supplied by
        SelfModel; reinit=True resamples immediately (relationship
        start).
        """
        for ax in THETA_AXES:
            if ax in prior:
                mu, sd = prior[ax]
                self._prior[ax] = (float(mu), float(sd))
        if reinit:
            self._init_particles()

    # ============================================================ Features
    @staticmethod
    def _feature_f(ctx: Optional[ObservationContext]) -> float:
        """f = reciprocity signal of my last action
        (+1 C / -1 D / 0 none)."""
        if ctx is None or ctx.my_last_action is None:
            return 0.0
        return 1.0 - 2.0 * float(ctx.my_last_action)

    @staticmethod
    def _feature_g(ctx: Optional[ObservationContext]) -> float:
        """g = the opponent's own last-action signal
        (+1 C / -1 D / 0 none)."""
        if ctx is None or ctx.their_last_action is None:
            return 0.0
        return 1.0 - 2.0 * float(ctx.their_last_action)

    # ============================================================ Likelihood
    def _pC(self, f: float, g: float) -> np.ndarray:
        """Per-particle predicted cooperation probability (N,)."""
        th = self.theta
        if self.es_provider is not None:
            shift = self.es_provider(th["lambda_j"], f, g)
        else:
            shift = empathy_shift(th["lambda_j"],
                                  self.my_cooperation_rate)
        logit = th["beta"] * (th["alpha"] + th["rho"] * f
                              + th["omega"] * g + th["eta"] * f * g
                              + shift)
        return _logistic(logit)

    def update(self, opp_action: int, ctx: ObservationContext) -> None:
        """
        Importance update with the observed opponent action
        (+ conditional resampling). The ctx tense must be the
        **update** tense (f = my_{k-2}, g = opp_{k-2}).
        """
        f = self._feature_f(ctx)
        g = self._feature_g(ctx)
        pC = self._pC(f, g)
        lik = pC if int(opp_action) == COOP else (1.0 - pC)

        self.weights = self.weights * np.clip(lik, _EPS, 1.0)
        s = float(self.weights.sum())
        if s <= _EPS:
            # No particle explains the observation -> restart from
            # the prior (numerical safeguard)
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights = self.weights / s

        ess = 1.0 / float(np.sum(self.weights ** 2))
        if ess < self.resample_frac * self.N:
            self._resample()
        self._h0_cache = None      # weights changed: invalidate H0 cache

    def _resample(self) -> None:
        """
        Systematic importance resampling + roughening.

        Resampling alone breeds duplicate particles and collapses the
        posterior into point masses (sample impoverishment). Per-axis
        jitter re-spreads the particles, and this artificial diffusion
        simultaneously implements the state-space assumption that
        **theta may drift slowly** — why trait-switch tracking (H1A)
        is possible.
        """
        idx = self.rng.choice(self.N, size=self.N, p=self.weights)
        for ax in THETA_AXES:
            lo, hi = _BOUNDS[ax]
            sd = self._JITTER[ax] * self.jitter_scale
            self.theta[ax] = np.clip(
                self.theta[ax][idx] + self.rng.normal(0.0, sd, self.N), lo, hi)
        self.weights = np.ones(self.N) / self.N

    # ============================================================ Prediction
    def predict_coop(self, f: float, g: float = 0.0) -> float:
        """Posterior-weighted cooperation probability P(a_j = C)."""
        return float(np.clip(np.sum(self.weights * self._pC(f, g)),
                             _EPS, 1.0 - _EPS))

    def predict_action(self, ctx: Optional[ObservationContext]) -> np.ndarray:
        """Opponent action distribution [P(C), P(D)] under decision-
        tense ctx."""
        pc = self.predict_coop(self._feature_f(ctx), self._feature_g(ctx))
        return np.array([pc, 1.0 - pc])

    # ============================================================ Summary stats
    def posterior_means(self) -> Dict[str, float]:
        """Posterior means per axis (SelfModel's theta)."""
        return {ax: float(np.sum(self.weights * self.theta[ax]))
                for ax in THETA_AXES}

    def posterior_stds(self) -> Dict[str, float]:
        """Posterior standard deviations per axis (SelfModel's
        dist)."""
        m = self.posterior_means()
        out = {}
        for ax in THETA_AXES:
            v = float(np.sum(self.weights * (self.theta[ax] - m[ax]) ** 2))
            out[ax] = float(np.sqrt(max(v, 0.0)))
        return out

    def reliability(self) -> float:
        """
        Weight-concentration reliability r in [0, 1] — the GatedToM
        gating signal. sqrt(ESS/N): 1 for uniform weights, 0 when mass
        collapses onto one particle.
        """
        ess = 1.0 / float(np.sum(self.weights ** 2))
        return float(np.clip(ess / self.N, 0.0, 1.0)) ** 0.5

    def belief_update_magnitude(self, prev_means: Dict[str, float]) -> float:
        """
        Standardised L2 norm of the change in posterior means — the
        scalar analogue of a model-based fMRI belief-update
        regressor.
        """
        cur = self.posterior_means()
        d = [(cur[ax] - prev_means.get(ax, cur[ax])) / self._PRIOR[ax][1]
             for ax in THETA_AXES]
        return float(np.sqrt(np.sum(np.square(d))))

    # ============================================================ Information gain
    # Epistemic term of the EFE. Discrete mutual information
    # I(a_j; theta) is approximated by the sum of per-axis marginal
    # histogram entropies; discreteness guarantees non-negativity.
    _HIST_NB = 11          # histogram bins
    _HIST_PAD = 0.5        # margin of the adaptive range (0.5 * SD)

    def _hist_entropy(self, arr: np.ndarray, w: np.ndarray, axis: str) -> float:
        """Shannon entropy of a weighted histogram."""
        m = float(np.sum(w * arr))
        sd = float(np.sqrt(max(float(np.sum(w * (arr - m) ** 2)), 1e-12)))
        if axis == "lambda_j":
            lo, hi = 0.0, 1.0                    # axis with a natural range
        else:
            lo, hi = m - 3.0 * sd - self._HIST_PAD, m + 3.0 * sd + self._HIST_PAD
        if hi - lo < 1e-9:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, self._HIST_NB + 1)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, self._HIST_NB - 1)
        b = np.zeros(self._HIST_NB)
        np.add.at(b, idx, w)
        s = b.sum()
        if s <= _EPS:
            return 0.0
        b = b / s
        nz = b[b > _EPS]
        return float(-np.sum(nz * np.log(nz)))

    def _total_entropy(self, w: np.ndarray) -> float:
        """Sum of marginal entropies over all axes."""
        return sum(self._hist_entropy(self.theta[ax], w, ax)
                   for ax in THETA_AXES)

    def _current_entropy(self) -> float:
        """
        All-axis entropy H0 of the current posterior. Repeatedly
        queried within a round while beliefs are unchanged, hence
        memoised; the weight array **object itself** is the cache key
        to avoid id-reuse bugs.
        """
        if self._h0_cache is not None and self._h0_cache[0] is self.weights:
            return self._h0_cache[1]
        h0 = self._total_entropy(self.weights)
        self._h0_cache = (self.weights, h0)
        return h0

    def expected_infogain_exact(self, f_next: float,
                                f_now: float, g_now: float) -> float:
        """
        [v3.7.5] Exact 1-step-ahead expected information gain —
        marginalising the opponent's simultaneous move.

        The frozen approximation (g_next = the opponent's last realised
        action) is removed: the opponent's **current-round** action
        a_j^t is marginalised under the posterior predictive of the
        current context. Conditioning on a_j^t changes two things at
        once:
          (i)  the next decision context g' = 1 - 2*a_j^t,
          (ii) the interim posterior update q(theta | a_j^t).
            IG(a) = sum_b p_now(b) * IG_next(f'(a), g'(b); q(theta|b))
        The information carried by a_j^t itself is action-independent
        (a common additive term) and is excluded — it cancels in the
        IG difference. Branches with p_now(b) ~ 0 are skipped
        (degeneracy guard).
        """
        pC_now = self._pC(f_now, g_now)
        w = self.weights
        p_now = float(np.sum(w * pC_now))
        total = 0.0
        for p_b, lik, g_b in ((p_now, pC_now, 1.0),
                              (1.0 - p_now, 1.0 - pC_now, -1.0)):
            if p_b < 1e-9:
                continue
            wb = w * lik
            sb = wb.sum()
            if sb <= 0.0:
                continue
            wb = wb / sb
            pC2 = self._pC(f_next, g_b)
            p_o = float(np.sum(wb * pC2))
            h0b = self._total_entropy(wb)

            def _post_h(l2):
                w2 = wb * l2
                s2 = w2.sum()
                return h0b if s2 <= 0.0 else self._total_entropy(w2 / s2)

            h_exp = p_o * _post_h(pC2) + (1.0 - p_o) * _post_h(1.0 - pC2)
            total += p_b * max(0.0, h0b - h_exp)
        return total

    def observed_infogain(self, obs_action: int, f: float,
                          g: float = 0.0) -> float:
        """
        **Realised** information gain — how much a specific observed
        action narrows this filter's posterior. Used by the
        self-projection filter: if I play obs_action (my own action is
        known with certainty), how much does the opponent's belief
        about me, theta-hat_self, tighten. "Realised", not
        "expected".
        """
        pC = self._pC(f, g)
        lik = pC if int(obs_action) == COOP else (1.0 - pC)
        H0 = self._current_entropy()
        w2 = self.weights * lik
        s = w2.sum()
        if s <= _EPS:
            return 0.0
        return max(0.0, H0 - self._total_entropy(w2 / s))
