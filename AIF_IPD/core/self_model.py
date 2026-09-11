"""
core.self_model
===============

**SelfModel — the memory store and prior supplier.**

Holds **(id, dist, theta, expected-reward distribution)** per
identity. Since v1.5.0 the expected-reward distribution is not the
marginal of observed rewards (the old QuantileCode) but a **quantile
vector derived from the QRTD values Z(s,a)** — "the long-run value
distribution of the situations I find myself in with this person".

The social reference of valence — a cross-population comparison
----------------------------------------------------------------
Premise: primary reward acquisition serves homeostasis over the
internal model, and the expected-reward distribution stands in as the
generative model of interoception. Valence is that model's subjective
fitness against **SelfModel's beliefs about the social environment**
— the expected-reward distributions of the other people inside the
self boundary:

    ref = sum_k w_k * Q_k / sum_k w_k    (k != current partner)
    w_k ~ inverse social distance (hyperbolic discounting)
    valence = 2 * F-hat_ref(median(Q_current)) - 1

  - The weighted mean of quantile vectors is the **Wasserstein
    barycentre** of the distributions, so ref precisely means "the
    typical value distribution of my social relationships".
  - The comparison is a **median rank**: the CDF position of the
    current partner's median within the reference.
  - That the reference lies in **other people, not in time** is
    decisive: temporal self-comparison lets the reference chase the
    current partner, extinguishing or inverting valence under chronic
    exploitation (measured); the population reference does not chase
    — however long the exploitation, the remembered others stay put,
    so "this relationship is bad relative to what I know" persists.
    That is allostasis.
  - The current partner is excluded from the population.

Pre-experiment social history — five people
----------------------------------------------------------------
About **5 relationships** inside the self boundary suffice (21 was
the quantile channel count, unrelated). Each person gets (id,
distance, cooperation rate, **value distribution**); value is
assigned directly with a cooperative-relationships-are-more-valuable
mapping (r-bar = 1 + 2c in [P, R] -> V = r-bar/(1-gamma)), and the
cooperation rate itself feeds only the distance-weighted setpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .distributional import DEFAULT_TAUS, vector_cdf

_EPS = 1e-12

THETA_AXES = ("alpha", "rho", "omega", "eta", "beta", "lambda_j")


#: Population moments of the distance-weighted (ratio - 0.5) under
#: the seeded social history (Monte-Carlo derived).
ALPHA_RATIO_MEAN = 0.02552757
ALPHA_RATIO_SD = 0.05500370


@dataclass
class MemoryEntry:
    """One partner's memory entry: (id, dist, theta, value
    distribution)."""
    identity: int
    theta: Dict[str, float] = field(default_factory=dict)
    theta_sd: Dict[str, float] = field(default_factory=dict)
    self_theta: Dict[str, float] = field(default_factory=dict)
    self_theta_sd: Dict[str, float] = field(default_factory=dict)
    # value_dist : the **expected-reward (value) distribution** — a
    # (21,) quantile vector. For experimental partners it is the
    # occupancy-EMA of Z(s,a); for seeded persons it is trained in
    # lockstep from r_bar below via the **same QR-Huber TD recursion**
    # (_tick_reference).
    value_dist: Optional[np.ndarray] = None
    # r_bar : characteristic per-round reward of a seeded person —
    # raw material for rolling the reference distribution through the
    # same estimation stage as the current partner's Z. None for
    # experimental partners.
    r_bar: Optional[float] = None
    # z_snapshot : full Z_self(s,a) (4,2,21) at the last commit — for
    # re-encounter restoration.
    z_snapshot: Optional[np.ndarray] = None
    familiarity: float = 0.0
    adv_self: Optional[float] = None    # occupancy-weighted dZ_self
    adv_other: Optional[float] = None   # occupancy-weighted dZ_other
    value_other: Optional[np.ndarray] = None  # policy-expected Z_other
    last_seen: int = 0
    n_obs: int = 0
    coop_count: float = 0.0


class SelfModel:
    """Identity memory + social distance + setpoints + population
    value reference."""

    def __init__(self,
                 theta_prior_mean: Optional[Dict[str, float]] = None,
                 theta_prior_std: Optional[Dict[str, float]] = None,
                 recency_tau: float = 200.0,
                 familiarity_scale: float = 30.0,
                 k_disc: float = 1.0,
                 prior_coop_weight: float = 1.0,
                 social_lr: float = 0.02,
                 identity_lr: float = 0.40,   # v2.8.0: earlier setpoint response
                 lam_floor: float = 0.0,
                 lam_ceil: float = 0.80,
                 taus=DEFAULT_TAUS):
        self.theta_prior_mean = dict(
            alpha=0.0, rho=0.5, omega=0.0, eta=0.0, beta=3.0, lambda_j=0.5)
        self.theta_prior_std = dict(
            alpha=2.0, rho=1.2, omega=1.0, eta=1.0, beta=1.8, lambda_j=0.3)
        if theta_prior_mean:
            self.theta_prior_mean.update(theta_prior_mean)
        if theta_prior_std:
            self.theta_prior_std.update(theta_prior_std)

        self.recency_tau = float(recency_tau)
        self.familiarity_scale = float(familiarity_scale)
        self.k_disc = float(k_disc)
        self.prior_coop_weight = float(prior_coop_weight)
        self.social_lr = float(social_lr)      # (reserved for reference drift)
        self.identity_lr = float(identity_lr)  # online EMA rate of value_dist
        self.lam_floor = float(lam_floor)
        self.lam_ceil = float(lam_ceil)
        self.taus = np.asarray(taus, dtype=float)

        self._payoff_span = 1.0
        self.memory: Dict[int, MemoryEntry] = {}
        self.clock: int = 0

    # ============================================================ Payoff scale
    def set_payoff_scale(self, payoffs: np.ndarray) -> None:
        """Record the payoff support span — used only as a
        normalisation constant."""
        u = np.asarray(payoffs, dtype=float)
        self._payoff_span = max(float(u.max() - u.min()), 1e-6)

    @property
    def payoff_span(self) -> float:
        return self._payoff_span

    # ============================================================ Social distance
    def social_distance(self, identity: Optional[int]) -> float:
        """d = 1/(1 + F/F_scale) in (0, 1]; unknown partners get 1."""
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None:
            return 1.0
        return float(1.0 / (1.0 + ent.familiarity / self.familiarity_scale))

    def distance_rank(self, identity: Optional[int]) -> float:
        """Unbounded ordinal distance N = F_scale / F (hyperbolic
        discounting applies here)."""
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or ent.familiarity <= _EPS:
            return float("inf")
        return float(self.familiarity_scale / ent.familiarity)

    def distance_weight(self, identity: Optional[int]) -> float:
        """Inverse-distance weight w = 1/(1 + k*N) — larger when
        closer (Jones & Rachlin)."""
        n = self.distance_rank(identity)
        if not np.isfinite(n):
            return 0.0
        return float(1.0 / (1.0 + self.k_disc * n))

    def _touch(self, identity: int) -> MemoryEntry:
        """Encounter tick: F <- F*exp(-dt/T) + 1 (frequency
        accumulation + recency forgetting)."""
        ent = self.memory.setdefault(identity, MemoryEntry(
            identity=int(identity), last_seen=self.clock))
        dt = max(self.clock - ent.last_seen, 0)
        ent.familiarity = ent.familiarity * float(
            np.exp(-dt / max(self.recency_tau, _EPS))) + 1.0
        ent.last_seen = self.clock
        self.clock += 1
        return ent

    def observe_identity(self, identity: int) -> None:
        self._touch(int(identity))

    # ============================================================ Prior supply
    def theta_prior(self, identity: Optional[int]) -> Dict[str, tuple]:
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or not ent.theta:
            return {ax: (self.theta_prior_mean[ax], self.theta_prior_std[ax])
                    for ax in THETA_AXES}
        out = {}
        for ax in THETA_AXES:
            mu = float(ent.theta.get(ax, self.theta_prior_mean[ax]))
            sd_floor = 0.25 * self.theta_prior_std[ax]
            sd = max(float(ent.theta_sd.get(ax, self.theta_prior_std[ax])),
                     sd_floor)
            out[ax] = (mu, sd)
        return out

    def self_theta_prior(self, identity: Optional[int]):
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or not ent.self_theta:
            return None
        return {ax: (float(ent.self_theta[ax]),
                     max(float(ent.self_theta_sd.get(ax, 0.3)), 0.15))
                for ax in ent.self_theta}

    def z_prior(self, identity: Optional[int]) -> Optional[np.ndarray]:
        """Z_self(s,a) snapshot to restore on re-encounter."""
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or ent.z_snapshot is None:
            return None
        return ent.z_snapshot.copy()

    # ============================================================ Memory updates
    def commit_theta(self, identity, theta, theta_sd) -> None:
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(identity=identity))
        ent.theta = {ax: float(theta.get(ax, 0.0)) for ax in THETA_AXES}
        ent.theta_sd = {ax: float(theta_sd.get(ax, 0.0)) for ax in THETA_AXES}

    def commit_self_theta(self, identity, theta, theta_sd) -> None:
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(identity=identity))
        ent.self_theta = {k: float(v) for k, v in theta.items()}
        ent.self_theta_sd = {k: float(v) for k, v in theta_sd.items()}

    def commit_observation(self, identity: Optional[int],
                           opponent_cooperated: Optional[bool] = None) -> None:
        """Accumulate whether the opponent cooperated — raw material
        of the distance-weighted cooperation rate (setpoint)."""
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(identity=identity))
        ent.n_obs += 1
        if opponent_cooperated is not None:
            ent.coop_count += 1.0 if opponent_cooperated else 0.0

    def update_partner_value(self, identity: Optional[int],
                             value_vector: np.ndarray,
                             z_snapshot: Optional[np.ndarray] = None,
                             value_other: Optional[np.ndarray] = None) -> None:
        """
        Online update of the current partner's **expected-reward
        (value) distribution**.

        value_vector is the Z quantile vector of the just-experienced
        (state, action); its EMA is the occupancy-weighted reduction
        "the value distribution of the situations I actually occupy
        with this person". An EMA of quantile vectors is a moving
        average in Wasserstein geometry, so the distributional meaning
        is preserved.
        """
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(identity=identity))
        if value_other is not None:
            vo = np.asarray(value_other, dtype=float)
            ent.value_other = (vo if ent.value_other is None
                               else (1.0 - self.identity_lr) * ent.value_other
                               + self.identity_lr * vo)
        v = np.asarray(value_vector, dtype=float)
        if ent.value_dist is None:
            ent.value_dist = v.copy()
        else:
            a = self.identity_lr
            ent.value_dist = (1.0 - a) * ent.value_dist + a * v
        if z_snapshot is not None:
            ent.z_snapshot = np.asarray(z_snapshot, dtype=float).copy()

    # ============================================================ Reference lockstep
    def value_prior(self) -> tuple:
        """
        **Uninformative prior for a new relationship** — (centre,
        spread), read from the value distributions of remembered
        relationships. Used to initialise the QRTD Z (legacy 'social'
        init; the current configuration uses the flat support-midpoint
        init instead), putting reference and Z on the same scale by
        construction. The principled origin is "a new relationship
        will resemble the ones I have lived", which is neither
        optimistic nor pessimistic; per-seed histories naturally
        individualise expectation levels.
        """
        rows = [e.value_dist for e in self.memory.values()
                if e.value_dist is not None and e.r_bar is not None]
        if not rows:
            return (2.0, 2.0)
        meds = np.array([r[len(r) // 2] for r in rows], dtype=float)
        widths = np.array([r[-1] - r[0] for r in rows], dtype=float)
        # Spread = the larger of between-relationship dispersion and
        # mean within-relationship width, preventing a reference so
        # narrow that valence saturates.
        spread = max(float(meds.std() * 2.0), float(widths.mean()), 1e-3)
        return (float(np.median(meds)), spread)

    # ============================================================ Population reference
    def population_reference(self, exclude: Optional[int] = None
                             ) -> Optional[np.ndarray]:
        """
        **Distance-weighted mean of the expected-reward distributions
        of others inside the self boundary.** A weighted mean of
        quantile vectors is a Wasserstein barycentre; the current
        partner (exclude) is left out. None if nobody remains (valence
        stays neutral 0).
        """
        num = None
        den = 0.0
        for pid, ent in self.memory.items():
            if pid == exclude or ent.value_dist is None:
                continue
            w = self.distance_weight(pid)
            if w <= _EPS:
                continue
            num = w * ent.value_dist if num is None else num + w * ent.value_dist
            den += w
        if num is None or den <= _EPS:
            return None
        return num / den

    def social_fitness(self, identity: Optional[int]) -> float:
        """
        **Between-relationship fitness** in (-1, 1) — how good the
        current partner is relative to remembered others; the source
        of the lambda setpoint:

            fitness = 2 * F-hat_ref(median(Q_current)) - 1
            ref = sum_k w_k * Q_k / sum_k w_k
                  (inverse-distance weights, current partner excluded)

        [v1.8.0 — separating the two comparisons] This value used to
        double as valence, but the between-relationship comparison
        answers "how far to open up" (a setpoint question) while the
        per-round affect answers "is this situation good within this
        relationship". Bound together, state-level prescriptions never
        reach affect — measured vs WSLS: the CD state demanded a
        defection detour (dZ = -3.88) yet cooperation stayed at 0.696;
        state-level valence flips the CD/DD ordering exactly opposite
        to TFT and revives the signal.
        """
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or ent.value_dist is None:
            return 0.0
        ref = self.population_reference(exclude=identity)
        if ref is None:
            return 0.0
        med = float(ent.value_dist[len(ent.value_dist) // 2])
        return float(2.0 * vector_cdf(ref, self.taus, med) - 1.0)

    # [social_valence removed] It was byte-identical to social_fitness
    # above, despite the v1.8.0 docstring describing the two as the
    # separated between-relationship and between-state comparisons. The
    # separation did happen, but the between-state half landed inline
    # in agent._regulate as `_st_val` (the occupancy-weighted state
    # percentile), so this copy answered no question of its own and had
    # no callers outside the tests.

    def reference_median(self, exclude: Optional[int] = None) -> float:
        ref = self.population_reference(exclude=exclude)
        return float(ref[len(ref) // 2]) if ref is not None else 0.0

    # ============================================================ Social history
    def seed_social_history(self, payoffs: np.ndarray,
                            gamma: float = 0.9,
                            within: float = 1.0,
                            n_close: int = 1, n_middle: int = 2,
                            n_far: int = 2,
                            coop_mean: float = 0.55, coop_sd: float = 0.18,
                            distance_coop_slope: float = 0.0,
                            rng: Optional[np.random.Generator] = None) -> None:
        """
        Seed the pre-experiment network of **five people**:
        (id, distance, cooperation rate, value distribution).

        - c_k ~ Beta(coop_mean, coop_sd) feeds only the setpoint —
          **no reward observations are generated**.
        - Value distributions are assigned directly:
          r-bar_k = 1 + 2c_k in [P, R], centre V_k on the per-round
          scale, width 30% of the value scale — the convention
          "cooperative relationships carry higher value".
        - distance_coop_slope defaults to 0: no distance-cooperation
          correlation is planted at initialisation.
        """
        rng = rng or np.random.default_rng(0)
        # v2.7.0: matching the Z-tilde normalisation, memory value
        # distributions are generated on the per-round mean-reward
        # scale; fitness is meaningful only if reference and Z share a
        # scale.
        v_scale = 1.0
        u = np.asarray(payoffs, dtype=float)
        self.set_payoff_scale(u)

        bands = [(n_close, 3.0), (n_middle, 0.6), (n_far, 0.1)]
        pid = -1
        m = float(np.clip(coop_mean, 0.02, 0.98))
        s2 = float(max(coop_sd, 1e-3)) ** 2
        nu = max(m * (1.0 - m) / s2 - 1.0, 0.1)

        for n_band, fam_mult in bands:
            for _ in range(int(n_band)):
                pid -= 1
                n_touch = max(int(round(fam_mult * self.familiarity_scale)), 1)
                for _ in range(n_touch):
                    self._touch(pid)
                d_k = self.social_distance(pid)
                c_k = float(rng.beta(m * nu, (1.0 - m) * nu))
                c_k = float(np.clip(
                    c_k + distance_coop_slope * (0.5 - d_k), 0.02, 0.98))

                ent = self.memory[pid]
                n_obs = max(int(round(0.5 * n_touch)), 2)
                ent.n_obs += n_obs
                ent.coop_count += float(round(c_k * n_obs))

                # **Assign the value distribution directly**
                # (v1.7.0 — the r-bar recursion was retired: the
                # reference is memory, not a learning target; rolling
                # it made all five converge and narrowed the
                # reference). Cooperative relationships sit higher:
                #   r-bar_k = 1 + 2c_k in [P, R], V_k = r-bar_k
                # Each person keeps their own width, so the weighted
                # reference retains **between-relationship dispersion**
                # for the current partner to rank against.
                # Advantage pair — memory holds no reciprocity
                # information, so derive from a memoryless
                # (state-independent) partner model with cooperation
                # probability c_k:
                #   dZ_self  = c(R-T) + (1-c)(S-P)   (< 0 in a PD)
                #   dZ_other = c(R-S) + (1-c)(T-P)   (> 0)
                pv = np.asarray(payoffs, dtype=float).reshape(-1)
                R_, S_, T_, P_ = pv[0], pv[1], pv[2], pv[3]
                ent.adv_self = c_k * (R_ - T_) + (1 - c_k) * (S_ - P_)
                ent.adv_other = c_k * (R_ - S_) + (1 - c_k) * (T_ - P_)
                ent.r_bar = 1.0 + 2.0 * c_k
                center = ent.r_bar * v_scale
                ent.value_dist = center + within * v_scale * (self.taus - 0.5)
                # The other's value level — what the partner got from
                # this relationship. If I was treated at rate c_k, the
                # partner symmetrically receives 1 + 2*(1-c_k) (the
                # PD's non-zero-sum asymmetry: my loss is their
                # gain).
                r_o = 1.0 + 2.0 * (1.0 - c_k)
                ent.value_other = (r_o * v_scale
                                   + within * v_scale * (self.taus - 0.5))

    def cooperation_bias(self, kappa: float = 2.0) -> float:
        """
        **Social-history cooperation bias alpha** (v2.2):

            ratio_k = Zbar_self^(k) / (Zbar_self^(k) + Zbar_other^(k))
            alpha = kappa * standardised distance-weighted
                    (ratio_k - 0.5)

        ratio = 0.5 (we gained equally) gives alpha = 0; having gained
        more than my partners pushes positive (cooperative approach),
        having been exploited pushes negative (defensive approach).
        Distance weights let close relationships dominate. Where
        lambda handles "this partner, this situation", alpha carries
        **"what social world do I come from"** — a social-indebtedness
        structure. kappa is in SD units: a 1-sigma history lands at
        alpha ~ +/- kappa on the logit scale.
        """
        num = den = 0.0
        for pid, ent in self.memory.items():
            if ent.value_dist is None or ent.value_other is None:
                continue
            zs = float(np.mean(ent.value_dist))
            zo = float(np.mean(ent.value_other))
            tot = zs + zo
            if abs(tot) < 1e-9:
                continue
            w = self.distance_weight(pid)
            num += w * (zs / tot - 0.5)
            den += w
        if den <= 1e-9:
            return 0.0
        # Standardise by the population dispersion: ALPHA_RATIO_SD is
        # the **population SD** of the distance-weighted (ratio - 0.5)
        # under the seeding distribution (coop_mean 0.55, sd 0.18),
        # computed once by a 2000-sample Monte Carlo. Hence
        # alpha ~ N(0, kappa^2) and kappa is literally the SD.
        z = (num / den - ALPHA_RATIO_MEAN) / ALPHA_RATIO_SD
        return float(kappa * z)

    def lambda_sp_compensatory(self, identity, adv_self, adv_other,
                               m: float = 0.20, i0: float = 0.0) -> tuple:
        """
        **Compensatory lambda setpoint** — baselined at the empathy
        this relationship requires:

            lam_sp = clip(lam*_j(I0) + m*Phi_j, 0, 1)
            lam*_j(I0) = (I0 - A_self)/(A_other - A_self)  (demand)
            Phi_j = 2*F-hat_ref(median(Q_j)) - 1           (disposition)

        Why the demand term is the baseline (v2.0): the fitness-affine
        map lam_sp = (1+Phi)/2 sets values **independently of the
        indifference point**, and defence collapsed in the lambda-only
        policy (vs ALLD: lambda 0.161 with indifference at 0.126 — a
        0.035 margin, intercept ~ 0, behaviour a coin flip). Anchoring
        at lam*_j(I0) starts lam_sp **near this relationship's
        threshold**, with m*Phi pushing good relationships up and bad
        ones down — sign is defined relative to the threshold, so both
        defence and cooperation are structural. A_self > 0
        (cooperation dominant on self-interest alone) sets the demand
        to 0.
        """
        if adv_other - adv_self <= 1e-9:
            base = 0.5
        elif adv_self > 0.0:
            base = 0.0          # cooperation dominant — no demand
        else:
            base = float(np.clip((i0 - adv_self) / (adv_other - adv_self),
                                 0.0, 1.0))
        phi = self.social_fitness(identity)
        return float(np.clip(base + m * phi, 0.0, 1.0)), float(phi), base

    # ============================================================ Setpoints
    def weighted_cooperation(self) -> float:
        num = self.prior_coop_weight * 0.5
        den = self.prior_coop_weight
        for pid, ent in self.memory.items():
            if ent.n_obs <= 0:
                continue
            w = self.distance_weight(pid)
            num += w * (ent.coop_count / ent.n_obs)
            den += w
        return float(num / max(den, _EPS))

    def lambda_setpoint(self, payoffs: Optional[np.ndarray] = None,
                        scale: float = 0.25) -> float:
        """lam0 = floor + (ceil-floor)*sigmoid((c-bar - 0.5)/scale);
        no memory -> 0.40."""
        c_bar = self.weighted_cooperation()
        sig = 1.0 / (1.0 + np.exp(-(c_bar - 0.5) / max(scale, 1e-6)))
        return float(self.lam_floor + (self.lam_ceil - self.lam_floor) * sig)

    # ============================================================ Diagnostics
    def social_summary(self) -> dict:
        rows = []
        for pid, ent in sorted(self.memory.items()):
            if ent.n_obs <= 0:
                continue
            rows.append({"id": pid, "distance": self.social_distance(pid),
                         "rank": self.distance_rank(pid),
                         "weight": self.distance_weight(pid),
                         "coop": ent.coop_count / max(ent.n_obs, 1),
                         "n_obs": ent.n_obs,
                         "value_median": (float(ent.value_dist[
                             len(ent.value_dist) // 2])
                             if ent.value_dist is not None else None)})
        return {"n_others": len(rows), "rows": rows,
                "weighted_cooperation": self.weighted_cooperation(),
                "lambda_setpoint": self.lambda_setpoint()}

    def snapshot(self, payoffs=None) -> dict:
        return {"n_identities": len(self.memory),
                "weighted_cooperation": self.weighted_cooperation(),
                "lambda_setpoint": self.lambda_setpoint(),
                "distances": {pid: self.social_distance(pid)
                              for pid in self.memory}}
