"""
core.qrtd
=========

**QRTD — quantile-regression temporal-difference learning
(Dabney et al. 2017, QR-DQN).**

Two objects live here:

  - `RewardModel` R-hat(joint) : the distribution of the **1-step
    reward**, supplying the learned likelihood intercept and per-step
    utilities.
  - `QuantileTD`  Z(s, a) : the **action-indexed return
    distribution** — a genuine distributional value function with TD
    bootstrapping, supplying expected-utility differences between
    action alternatives.

Why it is needed — when the payoff matrix is unknown:
`empathy_shift(lam, p)` computes the utility gap analytically **from
the payoff matrix**, which requires knowing (R, T, S, P). Earlier
versions privileged HalloReg with the current payoff matrix every
round even in non-stationary regimes, while fixed strategies received
nothing — so part of the H4 advantage could have been **information
privilege** rather than model quality. QRTD removes that confound:
learning Z(s, a) from observed rewards alone yields

    d-hat_lam(s) = (1-lam)*[V_self(s,C) - V_self(s,D)]
                 +   lam  *[V_other(s,C) - V_other(s,D)]

retrospectively, so alternatives can be compared without payoff
knowledge — supporting the stronger claim that the advantage holds
even when the payoff structure must be learned. `empathy_shift`
remains as the **closed-form oracle**, and the ARCH checks verify
that the learned d-hat converges to it.

Division of labour — avoiding double counting: Z is a discounted
return, so its differences **already contain the future**; a rollout
that re-accumulates the future would double-count. Within the finite
horizon the model rolls R-hat explicitly, and beyond it Z summarises:
G(theta_i) = -E[sum_{t<H} u_t + gamma^H * Z(s_H)] — the standard
model-based bootstrapped-planning split.

Value queries are **expectations**; the earlier risk-sensitive
reduction (CVaR_tau) was retired (see mean_value).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .distributional import DEFAULT_TAUS
from .constants import joint_index

_EPS = 1e-12

#: State = (my last action, their last action) -> 4; actions {C, D}.
N_STATES = 4
N_ACTIONS = 2


def state_index(my_last: int, opp_last: int) -> int:
    """s = 2*my_last + opp_last  (C=0, D=1)."""
    return 2 * int(my_last) + int(opp_last)


def mean_value(quantiles: np.ndarray) -> float:
    """
    Expectation of the distribution — on the midpoint grid the plain
    channel mean is exactly the midpoint-rule integral.

    [tau_risk retired — v1.5.3] The earlier lower-tail mean
    (CVaR_tau) was dropped because (1) self-protection is already
    lambda's job — imposing risk aversion again through tau lets the
    same construct enter twice and blurs lambda's interpretation; and
    (2) it is indistinguishable from estimation bias — under-visited
    cells linger near their init, and reading only the lower tail
    misreads that unlearned state as "risky" (measured amplification
    in naive mode). The distributional representation still earns its
    keep through valence (population rank) and arousal (W1 shift);
    value queries simplify to the expectation.
    """
    return float(np.mean(quantiles))


class _QuantileArray:
    """Shared QR-Huber update over a bundle of quantile vectors."""

    def __init__(self, shape: Tuple[int, ...], taus, init: float = 0.0,
                 spread: float = 1.0, lr: float = 0.05, kappa: float = 1.0):
        self.taus = np.asarray(taus, dtype=float)
        self.n = len(self.taus)
        self.lr = float(lr)
        self.kappa = max(float(kappa), 1e-6)
        # Shape (..., n); spread linearly in tau to define the
        # initial ordering.
        base = float(init) + float(spread) * (self.taus - 0.5)
        self.values = np.broadcast_to(base, shape + (self.n,)).copy()

    def set_scale(self, span: float) -> None:
        """
        Fit the Huber threshold kappa to the target's support range.

        kappa marks "errors beyond this size are outliers; saturate
        the step". With the normalised gradient kappa sets only the
        **loss curvature**, while the maximum per-update movement is
        bounded separately by lr*w — the two parameters' roles are
        decoupled.
        """
        self.kappa = max(0.25 * float(span), 1e-6)

    def _apply(self, idx: tuple, target: np.ndarray,
               lr: Optional[float] = None) -> None:
        """
        Apply a quantile Huber update to one (state, action) cell.

        target is a **distribution** (n samples): each target element
        creates loss against every quantile channel — the QR-DQN
        cross loss.
        """
        a = self.lr if lr is None else float(lr)
        cur = self.values[idx]                      # (n,)
        # delta[i, j] = target_j - theta_i
        delta = target[None, :] - cur[:, None]      # (n, n)
        w = np.where(delta > 0.0, self.taus[:, None], 1.0 - self.taus[:, None])
        # **Standard QR-Huber gradient** (v2.7.0), normalised by
        # kappa and hence bounded in [-1, 1]:
        #     d/d-delta [L_kappa(delta)/kappa] = clip(delta,-k,k)/k
        # The earlier unnormalised clip existed because raw Z lives on
        # a 1/(1-gamma) scale where the normalised form converged too
        # slowly, but it broke the Dabney et al. contraction condition
        # (alpha*kappa = 2.5 > 1). v2.7.0 normalises **Z itself**
        # (Z-tilde = (1-gamma)Z, mean per-round return), shortening
        # travel distances tenfold, so the standard normalised Huber
        # reaches the same speed (numerical check: 94.5% vs 92.6%
        # attainment after 600 updates).
        step = np.clip(delta, -self.kappa, self.kappa) / self.kappa
        upd = a * np.mean(w * step, axis=1)         # mean over target samples
        self.values[idx] = np.maximum.accumulate(cur + upd)


class QuantileTD:
    """
    **Z(s, a) — the distributional return value function**, learned
    in two copies (self and other): both payoffs are observed in the
    IPD, so the other's values are learnable too — supplying the
    lambda-weighted term of d-hat_lam.

    Parameters
    ----------
    gamma : float
        Return discount (0.9 in the current configuration; Z-tilde
        normalisation keeps values on the per-round reward scale).
    lr : float
        Quantile learning rate.
    init, spread : float
        Initial value and width of the quantile vectors, exposed as
        `init_value` / `init_spread` so SelfModel's reference
        distribution can share the same origin (scale alignment).
    """

    def __init__(self, taus=DEFAULT_TAUS, gamma: float = 0.9,
                 lr: float = 0.20,
                 init: float = 2.0, spread: float = 2.0,
                 seed: Optional[int] = None):
        self.taus = np.asarray(taus, dtype=float)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        shape = (N_STATES, N_ACTIONS)
        #: Observation-count shrinkage constant n0 (None = off; the
        #: current configuration keeps it off, so bvec == raw table).
        #:    bvec(s,a) = w*Z(s,a) + (1-w)*rbar_rel, w = n/(n+n0)
        self.shrink_n0 = None
        self.rbar = {"self": float("nan"), "other": float("nan")}
        self.init_value = float(init)
        self.init_spread = float(spread)
        #: Per-(state, action) visit counts — the denominator of the
        #: sample-mean learning rate.
        self.n_sa = np.zeros((N_STATES, N_ACTIONS), dtype=float)
        self.lr_base = float(lr)
        self.bootstrap = "sarsa"   # 'sarsa' | 'greedy'
        self.z_self = _QuantileArray(shape, taus, init=init, spread=spread,
                                     lr=lr)
        self.z_other = _QuantileArray(shape, taus, init=init, spread=spread,
                                      lr=lr)
        self.n_obs = 0

    def set_scale(self, span: float) -> None:
        """
        Set the Huber threshold on the **reward scale** (v2.7.0):
        Z-tilde = (1-gamma)Z is a per-round mean return, so kappa
        needs no 1/(1-gamma) inflation and the standard normalised
        Huber applies unchanged.
        """
        vspan = float(span)
        self.z_self.set_scale(vspan)
        self.z_other.set_scale(vspan)

    def reinit(self, center: float, spread: float) -> None:
        """
        Re-initialise Z from (center, spread).

        Called by the agent with the flat init: every Z(s,a) starts at
        the payoff-support midpoint c0 with a small tau-fan spread, so
        the reference (remembered relationships) and Z (the current
        relationship) begin on the same scale by construction and the
        first-round valence is neutral.
        """
        self.init_value = float(center)
        self.init_spread = float(spread)
        self.n_sa[:] = 0.0
        for arr in (self.z_self, self.z_other):
            base = float(center) + float(spread) * (arr.taus - 0.5)
            arr.values = np.broadcast_to(
                base, arr.values.shape[:-1] + (arr.n,)).copy()

    # [init_values / value_support removed] Neither had callers: the
    # agent initialises Z through reinit() with the flat support
    # midpoint (v3.7.3). Both were also still on the pre-v2.7.0
    # un-normalised scale, returning r/(1-gamma) -- about 10x too large
    # for Z-tilde at gamma=0.9 -- so reviving either would have
    # silently mis-scaled Z.

    # ============================================================ Update
    def update(self, s: int, a_i: int, r_self: float, r_other: float,
               s_next: int, a_next: int) -> Tuple[float, float]:
        """
        One QRTD update from the transition
        (s, a_i) -> (r_self, r_other, s', a').

        **True SARSA (v2.6.0)** — a' is not sampled: the caller holds
        the transition for one round and passes the action **actually
        chosen** at s'. Earlier versions sampled a' ~ Bern(q_c) with
        q_c produced at the *previous* state (a one-state-stale
        policy; measured state-conditional systematic error, worst for
        TFT). Knowing q_c(s') needs lambda_t, which needs the updated
        Z, which needs q_c(s') — holding one round observes the real
        a' and breaks the cycle **structurally**, also eliminating the
        target's sampling variance (Z_self and Z_other share the one
        real action automatically).

        Returns (td_target, shift) — CoreAffect's valence/arousal
        inputs.
        """
        # **Bootstrap mode** (v3.0)
        #   'sarsa'  : a' = the observed next action (on-policy)
        #   'greedy' : a' = argmax_a' Z_self(s', a')
        # greedy models "I know the best alternative without acting";
        # sarsa models "I only know the value of what I am doing".
        if self.bootstrap == "greedy":
            a_next = int(np.argmax([
                float(np.mean(self.z_self.values[s_next, 0])),
                float(np.mean(self.z_self.values[s_next, 1]))]))
        else:
            a_next = int(a_next)
        # **Normalised return** Z-tilde = (1-gamma)Z, so the reward
        # enters scaled by (1-gamma):
        #     Z(s,a) = (1-gamma)r + gamma*Z(s',a')
        # The fixed point is the mean per-round reward — same [0, 5]
        # scale as rewards.
        self.note_reward(r_self, r_other)
        _w_r = 1.0 - self.gamma
        tgt_s = (_w_r * float(r_self)
                 + self.gamma * self.bvec(s_next, a_next, "self"))
        tgt_o = (_w_r * float(r_other)
                 + self.gamma * self.bvec(s_next, a_next, "other"))
        prev = self.z_self.values[s, int(a_i)].copy()

        # **Visit-count adaptive learning rate** (v1.9.0)
        #   lr_eff(s,a) = max(1/(1+n(s,a)), lr_base)
        # The first visit lands on the target at lr = 1, then decays
        # as a sample mean toward the base rate (Robbins-Monro). This
        # removes init-value contamination: with a fixed lr, rarely
        # chosen cells linger at their init — a backdoor optimistic
        # initialisation (measured vs ALLD: A_self at 1/8 of theory).
        # One bad experience must immediately lower that action's
        # expectation.
        n_sa = self.n_sa[s, int(a_i)]
        lr_eff = max(1.0 / (1.0 + n_sa), self.lr_base)
        self.z_self._apply((s, int(a_i)), tgt_s, lr=lr_eff)
        self.z_other._apply((s, int(a_i)), tgt_o, lr=lr_eff)
        self.n_sa[s, int(a_i)] += 1.0
        self.n_obs += 1

        # Two quantities for CoreAffect:
        #   td_target : current reward + next-state value — a scalar
        #               summary of "how good was what just happened,
        #               implications included".
        #   shift     : W1(Z_before, Z_after) — how much the belief in
        #               this cell's long-run value moved; the arousal
        #               driver (the prospect, not the immediate reward,
        #               is what surprises).
        td_target = float(np.mean(tgt_s))
        shift = float(np.mean(np.abs(self.z_self.values[s, int(a_i)] - prev)))
        return td_target, shift

    # ============================================================ Queries
    def note_reward(self, r_self: float, r_other: float) -> None:
        """Update the relationship's empirical mean reward rbar_rel.
        Called by update() on the 1-step path and by the agent per
        round on the n-step path."""
        for _k, _r in (("self", r_self), ("other", r_other)):
            if not np.isfinite(self.rbar[_k]):
                self.rbar[_k] = float(_r)
            else:
                _b = max(1.0 / (1.0 + self.n_obs), 0.20)
                self.rbar[_k] += _b * (float(_r) - self.rbar[_k])

    def apply_target(self, s: int, a_i: int, tgt_s: np.ndarray,
                     tgt_o: np.ndarray) -> None:
        """Regress a cell directly onto an n-step target distribution
        (same rules as the 1-step update: visit-adaptive lr, QR-Huber;
        n_sa and n_obs counted identically)."""
        lr_eff = max(1.0 / (1.0 + float(self.n_sa[int(s), int(a_i)])),
                     self.lr_base)
        self.z_self._apply((int(s), int(a_i)), np.asarray(tgt_s, float),
                           lr=lr_eff)
        self.z_other._apply((int(s), int(a_i)), np.asarray(tgt_o, float),
                            lr=lr_eff)
        self.n_sa[int(s), int(a_i)] += 1
        self.n_obs += 1

    def bvec(self, s: int, a: int, which: str = "self") -> np.ndarray:
        """Bootstrap quantile vector — shrunk toward rbar_rel when
        shrinkage is enabled (off in the current configuration)."""
        arr = self.z_self if which == "self" else self.z_other
        v = arr.values[int(s), int(a)]
        if self.shrink_n0 is None or not np.isfinite(self.rbar[which]):
            return v
        w = float(self.n_sa[int(s), int(a)]) / (
            float(self.n_sa[int(s), int(a)]) + float(self.shrink_n0))
        return w * v + (1.0 - w) * self.rbar[which]

    def value(self, s: int, a: int, which: str = "self",
              tau: Optional[float] = None) -> float:
        """Expected value V-hat(s, a); tau is signature compat,
        ignored."""
        return float(np.mean(self.bvec(s, a, which)))

    def terminal_value(self, s: int, p_coop: float, lam: float) -> float:
        """
        Lambda-weighted value of a terminal state, marginalised over
        actions by the policy cooperation probability — the summary of
        everything beyond the rollout horizon.
        """
        v = 0.0
        for a, pa in ((0, float(p_coop)), (1, 1.0 - float(p_coop))):
            if pa <= _EPS:
                continue
            v += pa * ((1.0 - lam) * self.value(s, a, "self")
                       + lam * self.value(s, a, "other"))
        return float(v)

    def shift(self, s: int, lam: float) -> float:
        """
        **d-hat_lam(s) = V(s,C) - V(s,D) — the return difference.
        Diagnostics/validation only.**

        Do not use this as the intercept of a rollout's per-step
        likelihood: the trait-EFE already counts the future once
        inside the horizon (explicit rollout) and once beyond it
        (terminal Z), so injecting the return difference into every
        step's intercept counts the future H+1 times. The intercept's
        job is the immediate consequence of this one move; the rest
        belongs to the rollout and the terminal term. Scale confirms
        it: at gamma = 0.9 the raw return difference ran ~40x the
        1-step analytic shift, instantly saturating the sigmoid.
        (This concern is specific to the legacy trait rollout; the
        lambda-only path uses the normalised Z-tilde advantage, where
        no rollout re-counts the future.)
        """
        ds = self.value(s, 0, "self") - self.value(s, 1, "self")
        do = self.value(s, 0, "other") - self.value(s, 1, "other")
        return float((1.0 - lam) * ds + lam * do)

    # ============================================ Planning sweeps (v3.0)
    def _mix_quantiles(self, va, vb, w):
        """
        Quantiles of the **true mixture** w*F_a + (1-w)*F_b (not a
        barycentre): treat both vectors as atoms weighted w/N and
        (1-w)/N, sort, accumulate weights, and interpolate onto the
        tau grid.
        The quantile average w*va + (1-w)*vb is a Wasserstein
        barycentre and flattens bimodality (e.g. a 30/50 mixture
        becomes all-channel 40), so it is not used.
        """
        n = va.shape[0]
        x = np.concatenate([va, vb])
        ww = np.concatenate([np.full(n, w / n), np.full(n, (1.0 - w) / n)])
        o = np.argsort(x)
        x = x[o]; ww = ww[o]
        cw = np.cumsum(ww) - 0.5 * ww
        return np.interp(self.taus, cw, x)

    def _conv_project(self, r_atoms: np.ndarray, v_atoms: np.ndarray,
                      w_r: float) -> np.ndarray:
        """
        **Full distributional Bellman target** — the law of
        (1-gamma)R + gamma*V (v3.3).

        Assuming R ~ R-hat (21 atoms) and V ~ the bootstrap
        distribution (21 atoms) independent, convolve: form 441
        equal-mass atoms (1-gamma)r_k + gamma*v_m and project onto the
        tau grid under the midpoint-accumulation convention. Feeding
        R-hat as a mean scalar (the old way) kept reward uncertainty
        out of the target's shape; the full form propagates R-hat's
        spread into Z-tilde exactly when non-stationary regimes make
        it real.
        """
        b = w_r * r_atoms[:, None] + self.gamma * v_atoms[None, :]
        x = np.sort(b.ravel())
        cw = (np.arange(x.size) + 0.5) / x.size
        return np.interp(self.taus, cw, x)

    def plan_sweep(self, rmodel, p_coop_j: float, p_coop_self: float,
                   n_sweeps: int = 1, update: str = "replace",
                   lr: float = 0.5) -> None:
        """
        **Distributional value iteration over the learned generative
        model** (a Dyna-style planning sweep):

            Z(s,a) <- mix_{a_j~Bern(p_j)}[(1-gamma)R-hat(a,a_j)
                        + gamma * mix_{a'~pi} Z(s'(a,a_j), a')]

        All 8 (s, a) cells update every round, so convergence is
        governed by the **gamma contraction**, not by sample visits
        (error shrinks 0.9x per sweep; within 1% by ~40 rounds).

        What it fixes (diagnosis): after 400R vs ALLD both Z(DD,.)
        cells sat ~+1.0 above their true on-policy values because
        (i) the old social-prior init was 3x this relationship's true
        value, (ii) the bootstrap targets themselves were
        contaminated, and (iii) visit imbalance left extra bias in the
        rarer cell — compressing the advantage to -0.034 vs the true
        -0.100, leaving 0.32 cooperation with an exploiter even at
        lambda = 0.017. The all-cell sweep removes all three at once:
        unvisited counterfactual branches get correct model values
        (p_j -> 0 evaluates them exactly) and the init is washed out
        by the contraction.

        Lineage: Sutton (1990) Dyna; Moore & Atkeson (1993)
        prioritized sweeping; a distributional Expected-SARSA
        (van Seijen et al. 2009). The a_j marginalisation uses **true
        mixture quantiles**, not barycentres, hence no bias.

        Cost: model trust transfers R-hat/p_j errors into values —
        immediately after a payoff-regime switch the sweep uses stale
        values, but R-hat relearns quickly under the visit-adaptive
        rate, so the window is short.
        """
        # p_coop_j is a **per-state vector (4,)**: a single scalar
        # erases the opponent's conditionality on my previous action
        # (TFT would be misrepresented as a random cooperator;
        # measured: TFT CC 0.850 -> 0.674).
        pj = np.clip(np.asarray(p_coop_j, dtype=float).reshape(-1), 0.0, 1.0)
        if pj.size == 1:
            pj = np.full(4, float(pj[0]))
        # p_coop_self may also be per-state (v3.3): my continuous
        # policy sigmoid(beta_es * es(lambda, s')) is state-dependent,
        # and a scalar would leave the same state-blind approximation
        # in planning that true SARSA removed from learning.
        pc = np.clip(np.asarray(p_coop_self, dtype=float).reshape(-1),
                     0.0, 1.0)
        if pc.size == 1:
            pc = np.full(4, float(pc[0]))
        w_r = 1.0 - self.gamma
        for _ in range(int(n_sweeps)):
            for arr, rvec in ((self.z_self, rmodel.r_self),
                              (self.z_other, rmodel.r_other)):
                # Next-state values marginalised under pi, (4,) x N
                nxt = np.stack([
                    self._mix_quantiles(arr.values[sp, 0],
                                        arr.values[sp, 1], float(pc[sp]))
                    for sp in range(4)])
                new = np.empty_like(arr.values)
                for s in range(4):
                    for a in range(2):
                        # Build both a_j branches, then true-mix
                        br = []
                        for aj in (0, 1):
                            j = joint_index(a, aj)
                            br.append(self._conv_project(
                                rvec.values[j], nxt[j], w_r))
                        new[s, a] = self._mix_quantiles(br[0], br[1],
                                                       float(pj[s]))
                if update == "conf":
                    # **Model-confidence-conditioned Dyna** — the
                    # planning rate is conditioned on how observed the
                    # model cells feeding this target are:
                    #   alpha(s,a) = p_j(s)*w(n_R[(a,C)])
                    #              + (1-p_j(s))*w(n_R[(a,D)]),
                    #   w(n) = n/(n+2).
                    # This fixes the failure of visit-based gating
                    # (contaminated cells are exactly the most-visited
                    # ones) by conditioning on the right variable:
                    # where the model is well estimated alpha -> 1
                    # (replacement — contamination removal kept),
                    # where it is uncertain the samples dominate.
                    nv = rmodel.n_visit
                    for s2 in range(4):
                        for a2 in range(2):
                            _wc = nv[joint_index(a2, 0)] / (
                                nv[joint_index(a2, 0)] + 2.0)
                            _wd = nv[joint_index(a2, 1)] / (
                                nv[joint_index(a2, 1)] + 2.0)
                            _lr = float(pj[s2] * _wc + (1.0 - pj[s2]) * _wd)
                            arr._apply((s2, a2), new[s2, a2],
                                       lr=max(_lr, 0.05))
                elif update == "dyna":
                    # **Symmetric Dyna** (exploratory) — planning
                    # uses the same visit-adaptive rate as direct
                    # learning: unvisited cells alpha ~ 1 (fast
                    # counterfactual correction), well-visited cells
                    # alpha = 0.20 (samples dominate).
                    for s2 in range(4):
                        for a2 in range(2):
                            _lr = max(1.0 / (1.0 + float(self.n_sa[s2, a2])),
                                      0.20)
                            arr._apply((s2, a2), new[s2, a2], lr=_lr)
                elif update == "td":
                    # **Incremental planning** (exploratory) — the
                    # quantile analogue of Z <- Z + alpha*(T_model Z -
                    # Z); all targets were computed from old values
                    # (synchronous), so per-cell regression preserves
                    # the Jacobi property.
                    for s2 in range(4):
                        for a2 in range(2):
                            arr._apply((s2, a2), new[s2, a2], lr=lr)
                else:
                    arr.values[:] = np.maximum.accumulate(new, axis=-1)


class RewardModel:
    """
    **R-hat(joint) — the distributional model of the 1-step
    reward.**

    Learns self/other reward quantiles for each of the 4 joint
    outcomes. Deterministic payoffs converge to point masses; in
    non-stationary regimes the distribution genuinely spreads and
    that width itself is information. This is what replaces
    PAYOFF_SELF / PAYOFF_OTHER — the component that frees the agent
    from privileged payoff access.
    """

    def __init__(self, taus=DEFAULT_TAUS, lr: float = 0.10,
                 init: float = 2.0,
                 spread: float = 2.0):
        self.taus = np.asarray(taus, dtype=float)
        self.r_self = _QuantileArray((4,), taus, init=init, spread=spread,
                                     lr=lr)
        self.r_other = _QuantileArray((4,), taus, init=init, spread=spread,
                                      lr=lr)
        self.n_obs = 0
        #: Per-joint-outcome observation counts — the decay factor of
        #: epistemic uncertainty.
        self.n_visit = np.zeros(4, dtype=float)

    def set_scale(self, span: float) -> None:
        self.r_self.set_scale(span)
        self.r_other.set_scale(span)

    def update(self, joint: int, r_self: float, r_other: float) -> None:
        """Update from an observed joint outcome; the target is a point
        (length-1 sample)."""
        self.r_self._apply((int(joint),), np.array([float(r_self)]))
        self.r_other._apply((int(joint),), np.array([float(r_other)]))
        self.n_visit[int(joint)] += 1.0
        self.n_obs += 1

    def payoff_vector(self, which: str = "self",
                      tau: Optional[float] = None) -> np.ndarray:
        """
        The learned (4,) payoff vector (expectation reduction) — a
        drop-in replacement for PAYOFF_SELF.
        """
        arr = self.r_self if which == "self" else self.r_other
        return np.array([mean_value(arr.values[j]) for j in range(4)],
                        dtype=float)

    def shift(self, lam: float, p_coop: float) -> float:
        """
        **s-hat(lambda, p) — the learned 1-step expected-utility
        difference**: the same formula as empathy_shift(lambda, p)
        computed from learned rather than true payoffs. Under
        stationary payoffs it must converge to the analytic solution;
        the ARCH checks verify this.
        """
        us = self.payoff_vector("self")
        uo = self.payoff_vector("other")
        p = float(np.clip(p_coop, 0.0, 1.0))
        # Joint index convention: CC=0, CD=1, DC=2, DD=3
        d_self = (p * us[0] + (1 - p) * us[1]) - (p * us[2] + (1 - p) * us[3])
        d_other = (p * uo[0] + (1 - p) * uo[1]) - (p * uo[2] + (1 - p) * uo[3])
        return float((1.0 - lam) * d_self + lam * d_other)

    def epistemic_gain(self) -> np.ndarray:
        """
        **Expected information gain per joint outcome (4,), in
        nats** — v3.7.2 (path A).

        Treating each cell's reward as a Normal likelihood with a
        Normal-inverse-gamma conjugate posterior, the expected KL of
        one observation about mu is closed-form:
            E_r[KL(q(mu|r) || q(mu))] = 1/2 ln(1 + 1/kappa),
            kappa = 1 + n_visit.
        The aleatoric variance cancels inside the log, so the
        epistemic/aleatoric split is **derived** rather than assumed
        (the principled form of the old spread/sqrt(1+n) heuristic;
        ~1/(2n) decay for large n). Same units (nats) as IG_theta, so
        equal weights w_R = w_theta are unit-consistent, not an
        arbitrary tuning.
        """
        return 0.5 * np.log1p(1.0 / (1.0 + self.n_visit.astype(float)))

    def uncertainty(self) -> np.ndarray:
        """
        **Epistemic uncertainty per joint outcome (4,)** — raw
        material of the environment-structure epistemic affordance:

            U_j = spread_j / sqrt(1 + n_j)

        Spread alone mixes reducible (epistemic) and irreducible
        (aleatoric) width; information gain must target only the
        former, otherwise IG_R never extinguishes and keeps luring
        probing defections after learning is done (measured 0.38 ->
        0.31 plateau; ablation showed harm only against strict
        reciprocators, where a single probe starts an echo chain).
        1/sqrt(1+n) is the standard-error decay of a sample mean, so
        exploration switches itself off; the spread factor stays
        because, at equal visits, a more variable cell genuinely is
        more uncertain — needed in non-stationary payoff regimes.
        """
        sp = np.array([float(self.r_self.values[j][-1]
                             - self.r_self.values[j][0]) for j in range(4)],
                      dtype=float)
        return sp / np.sqrt(1.0 + self.n_visit)

    def snapshot(self) -> dict:
        return {"payoff_self": self.payoff_vector("self").tolist(),
                "payoff_other": self.payoff_vector("other").tolist(),
                "n_obs": int(self.n_obs)}
