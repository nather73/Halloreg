"""
ipd.agent
=========

Active-inference IPD agents.

  EmpathicAgent : Reproduction of Albarracin et al. (2026): gated ToM
                  particle filter + recursive social EFE, with an
                  **exogenous, fixed lambda**. It inherits the paper's
                  limitation (a high baseline lambda cannot self-protect
                  against exploiters) and serves both as the control
                  condition and as the "opponent with a known lambda"
                  in the lambda-recovery experiment (H1A).

  HalloRegAgent : **The proposed model of this project.** It adds
                  hierarchical allostatic regulation on top of
                  EmpathicAgent (SelfModel -> CoreAffect /
                  OpponentInversion -> Empathy) so that lambda is
                  produced endogenously.

[Per-round information flow of HalloRegAgent]

    SelfModel ──(id, theta)──────▶ OpponentInversion ──┐ p_j(s), IG_θ
        │                                               │
        ├──(id, value dist, Z snapshot)──▶ QuantileTD ──┤ Z̃(s,a)
        │                                               ▼
        │                                   allostatic mapping (_regulate)
        │                                        λ = clip(λ*(s) + φ(s) − ½)
        │                                               │
        └──(baseline value dist)                        ▼
                                          RecursiveSocialEFE → action

        CoreAffect ──(valence, arousal, λ_aff)──▶ [logged only]

    **CoreAffect is not currently wired into the lambda path.** This
    matches the architecture document §2.5 ("CoreAffect is not yet
    linked with other components"): valence, arousal and λ_aff are
    computed every round and recorded for analysis, but no term of
    them enters the lambda update in `_regulate`. ARCH check V7
    asserts the decoupling directly — running a dyad with the affect
    outputs pinned to zero must leave the lambda trajectory
    bit-identical — so the situation cannot drift out of sync with the
    document again.

    The `Empathy` integrator is retired for the same reason: since the
    allostatic direct mapping replaced it, `Empathy.step` has no call
    sites anywhere in the package and the object survives only as the
    carrier of `.lam`.

    The updated (theta, dist) and (expected reward dist) are committed
    back into SelfModel: SelfModel does no inference — it **only
    remembers**.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from AIF_IPD.core.constants import (
    COOP, DEFECT, PAYOFF_SELF, empathy_shift, opponent_action_from_state,
)
from AIF_IPD.core.core_affect import CoreAffect
from AIF_IPD.core.empathy import Empathy
from AIF_IPD.core.constants import joint_index
from AIF_IPD.core.qrtd import QuantileTD, RewardModel, state_index
from AIF_IPD.core.self_model import SelfModel
from .tom.self_policy import SELF_AXES, SelfPolicy
from AIF_IPD.core.empathy import empathy_shift_z
from .tom import (
    GatedToM, ObservationContext, OpponentInversion, RecursiveSocialEFE,
    TheoryOfMind,
)


class EmpathicAgent:
    """
    Fixed-lambda empathic agent (paper reproduction + recursive
    extension).

    Parameters
    ----------
    lam : float
        Fixed empathy weight, lambda in [0, 1].
    beta_self, beta_other : float
        Action-selection precision for self / opponent.
    n_particles : int
        Particle count of the opponent theta-hat filter.
    planning_horizon : int
        Rollout horizon H in **trait space**; 1 = myopic.
    recursive_depth : int
        2 enables depth-2 perspective taking (self-projection filter).
    self_projection : bool
        Whether to run the self-projection filter theta-hat_self.
        False disables IG_other and depth-2.
    """

    #: Logging channels — subclasses extend this list.
    #: Action-selection mode, overridden by subclasses (EmpathicAgent
    #: uses the legacy trait path).
    policy_mode = "traits"
    beta_es = 45.0

    LOG_KEYS = ("action", "lam", "pred_coop", "reliability",
                "S_rho", "S_omega", "S_eta", "SD_S_rho", "SD_S_omega",
                "SD_S_eta", "policy_ess", "policy_G", "q_coop",
                "E_alpha", "E_rho", "E_omega", "E_eta", "E_beta", "E_lambda_j",
                "SD_alpha", "SD_rho", "SD_omega", "SD_eta", "SD_beta",
                "SD_lambda_j", "belief_update", "self_pred_coop",
                # Posterior means of the self-projection filter
                # theta-hat_self — "this is how the opponent will infer
                # me". Used by H6 projection-consistency checks.
                "P_alpha", "P_rho", "P_omega", "P_eta", "P_beta",
                "P_lambda_j")

    def __init__(self, lam: float = 0.4, beta_self: float = 4.0,
                 policy_particles: int = 64, prop_sd: float = 0.40,
                 policy_gamma: float = 8.0,
                 w_epi_j: float = 10.0, w_epi_r: float = 1.0,
                 w_cplx: float = 0.15,
                 beta_other: float = 4.0, n_particles: int = 400,
                 planning_horizon: int = 6, recursive_depth: int = 2,
                 self_projection: bool = True, jitter_scale: float = 1.0,
                 lam_schedule: Optional[list] = None,
                 name: str = "Empathic", seed: int = 0):
        self.name = name
        self.lam = float(lam)
        self.planning_horizon = int(planning_horizon)
        # [H1A] Lambda switch schedule [(round, lam), ...]: builds an
        # opponent whose known lambda changes mid-session, to test that
        # the inverter tracks the **change** in lambda-hat_j.
        self.lam_schedule = sorted(lam_schedule or [], key=lambda x: x[0])
        self.rng = np.random.default_rng(seed + 991)

        # ---- Perspective-taking stack ----
        self.inversion = OpponentInversion(n_particles=n_particles,
                                           jitter_scale=jitter_scale,
                                           seed=seed)
        # Self-projection filter: apply the **same inverter** to my own
        # action history to construct "how the opponent will infer me".
        # Half the particles suffice (my own actions are known, so the
        # posterior is narrow).
        self.self_inversion = (
            OpponentInversion(n_particles=max(n_particles // 2, 200),
                              jitter_scale=jitter_scale, seed=seed + 7)
            if self_projection else None)

        self.tom = TheoryOfMind(beta_other=beta_other)
        self.gated = GatedToM(self.tom, self.inversion)
        # Trait-space policy (legacy EmpathicAgent action path).
        self.self_policy = SelfPolicy(
            n_particles=policy_particles, prop_sd=prop_sd, gamma=policy_gamma,
            horizon=planning_horizon, w_epi_j=w_epi_j, w_epi_r=w_epi_r,
            w_cplx=w_cplx,
            beta_self=beta_self,
            seed=(None if seed is None else seed * 7717 + 31))
        self._last_policy = {ax: 0.0 for ax in SELF_AXES}
        self._last_qc = 0.5

        self.social_efe = RecursiveSocialEFE(
            self.gated, self.inversion, empathy_factor=self.lam,
            beta_self=beta_self, recursive_depth=recursive_depth,
            self_inversion=self.self_inversion)

        # ---- Interaction state ----
        # EmpathicAgent (control) has no QRTD: it is assumed to know the
        # payoff matrix and always uses the analytic s(lambda, p);
        # HalloRegAgent overrides this.
        self.payoff_access = "oracle"
        self.reward_model = None
        self.qrtd = None
        self._prev_sa = None
        #: Pending SARSA transitions (s, a, r_self, r_other, s_next),
        #: applied once a' is observed.
        self._pending = []
        self._shift_used = float("nan")
        self._g_prag = self._g_epi_r = self._g_epi_j = float("nan")
        self._shift_oracle = float("nan")

        self.my_last = COOP                 # my previous action (init: C)
        self.my_actions: List[int] = []
        self.opp_actions: List[int] = []    # observed opponent history
        self._prev_means = self.inversion.posterior_means()

        self.log: Dict[str, list] = {k: [] for k in self.LOG_KEYS}

    # ============================================================ Hooks
    def begin_partner(self, identity: int) -> None:
        """
        Partner-identity notification (called by the environment).
        Fixed-lambda agents keep no partner memory, so it is a no-op.
        """
        return None

    def note_emitted(self, emitted: int) -> None:
        """
        **Execution-error reconciliation** (called by the environment).

        run_dyad may stochastically flip the action the agent chose. The
        opponent reacts to the **actually emitted** action, so the
        agent's internal `my_last` (the source of next round's
        reciprocity signal f) must be set to the emitted action as
        well. Otherwise the likelihood's f diverges from what the
        opponent actually saw (a value mismatch, not an off-by-one) and
        the reciprocity estimate rho-hat collapses systematically.

        The same holds for the self-projection filter — the opponent
        observed the emitted action. Since update() has already run, we
        only correct the state used by later rounds (retroactive
        correction of past particle updates is impossible, and with low
        error rates its effect is negligible).
        """
        e = int(emitted)
        if self.my_actions:
            self.my_actions[-1] = e
        self.my_last = e

    def current_lambda(self) -> float:
        """
        Lambda used for action selection. Constant by default; with a
        lam_schedule, switch to the entry reached at the current round
        (the last reached entry wins).
        """
        if self.lam_schedule:
            t = len(self.my_actions)
            for r, v in self.lam_schedule:
                if t >= r:
                    self.lam = float(v)
            self.social_efe.lam = self.lam
        return self.lam

    def _occ_adv(self):
        """Occupancy-weighted action advantages (A_self, A_other) —
        input of the compensatory lambda setpoint."""
        if self.qrtd is None or self.qrtd.n_obs <= 3:
            return None
        d = self._state_visits / max(self._state_visits.sum(), 1e-9)
        a_s = float(np.sum([d[ss] * (self.qrtd.value(ss, COOP, "self")
                                     - self.qrtd.value(ss, DEFECT, "self"))
                            for ss in range(4)]))
        a_o = float(np.sum([d[ss] * (self.qrtd.value(ss, COOP, "other")
                                     - self.qrtd.value(ss, DEFECT, "other"))
                            for ss in range(4)]))
        return (a_s, a_o)

    def _tom_mirror_es(self, lam_vec, f, g):
        """Mirror es_z provider (v3.7.1), called by
        OpponentInversion._pC."""
        if f == 0.0 or g == 0.0:
            from AIF_IPD.core.constants import empathy_shift
            return empathy_shift(lam_vec,
                                 self.inversion.my_cooperation_rate)
        my_a = 0 if f > 0 else 1
        th_a = 0 if g > 0 else 1
        m = joint_index(th_a, my_a)        # state from their view (mirror)
        a_s = (self.qrtd.value(m, 0, "self")
               - self.qrtd.value(m, 1, "self"))
        a_o = (self.qrtd.value(m, 0, "other")
               - self.qrtd.value(m, 1, "other"))
        return (self.w_u
                * ((1.0 - lam_vec) * a_s + lam_vec * a_o))

    def _regulate(self, observed_state: int, opp_action: int,
                  inferred: dict) -> dict:
        """Lambda-regulation hook; no-op here, overridden by
        HalloRegAgent."""
        return {}

    # ============================================================ One round
    def step(self, observed_state: Optional[int]) -> int:
        """
        Observe the joint outcome of round t-1 and return the action
        for round t. observed_state=None means the first round (no
        observation yet).
        """
        # ---------- (0) Decision context ----------
        # The prediction target is opp_k; the opponent reacts to the
        # latest information (my_{k-1}, opp_{k-1}).
        ctx = ObservationContext(
            my_last_action=self.my_last,
            their_last_action=(self.opp_actions[-1] if self.opp_actions else None),
            round_number=len(self.my_actions))

        reg = {}
        belief_update = 0.0

        if observed_state is not None:
            opp_action = opponent_action_from_state(observed_state)
            ctx.their_last_action = opp_action
            ctx.joint_outcome = observed_state
            self._prev_means = self.inversion.posterior_means()

            # ---------- (1) Opponent theta-hat filter update ----------
            # Update-time tense: f = my_{k-2}, g = opp_{k-2}.
            # opp_actions has NOT yet appended opp_{k-1}, so
            # opp_actions[-1] is exactly opp_{k-2}. (The append happens
            # below — order-dependent, never reorder.)
            f_upd = self.my_actions[-2] if len(self.my_actions) >= 2 else None
            g_upd = self.opp_actions[-1] if self.opp_actions else None
            ctx_upd = ObservationContext(
                my_last_action=f_upd, their_last_action=g_upd,
                joint_outcome=observed_state, round_number=len(self.my_actions))
            self.inversion.update(opp_action, ctx_upd)

            inferred = self.inversion.posterior_means()
            belief_update = self.inversion.belief_update_magnitude(self._prev_means)

            # ---------- (2) Lambda regulation (subclass) ----------
            reg = self._regulate(observed_state, opp_action, inferred)

            # ---------- (3) Record opponent action ----------
            self.opp_actions.append(opp_action)

        # ---------- (4) Policy -> action ----------
        # Note the basis-role swap: in the focal policy
        #   f = the opponent's last action (what I reciprocate)
        #   g = my own last action (self-inertia)
        # whereas in the opponent model f was "my last action".
        lam = self.current_lambda()
        theta_j = self.inversion.posterior_means()

        f_me = (0.0 if not self.opp_actions
                else 1.0 - 2.0 * float(self.opp_actions[-1]))
        g_me = (0.0 if self.my_last is None
                else 1.0 - 2.0 * float(self.my_last))

        # Opponent coop rate p_j (s(lambda, p) argument of my
        # likelihood) and my coop rate p_i (for their likelihood).
        p_other = (float(np.mean([1.0 - a for a in self.opp_actions]))
                   if self.opp_actions else 0.5)
        p_self = (float(np.mean([1.0 - a for a in self.my_actions]))
                  if self.my_actions else 0.5)

        # ---- The two epistemic terms ----
        # (a) IG_j : information gain about the opponent's hidden
        #     intent theta-hat_j (per action)
        # (b) IG_R : information gain about the **latent environment
        #     structure R-hat** (per action)
        # My action changes which joint outcomes I will observe, so an
        # action that visits an unexperienced cell (e.g. DC in a locked
        # cooperative relationship) earns high IG. Exploration comes
        # from the objective, not from a prior.
        p_coop_j = float(self.inversion.predict_coop(g_me, f_me))
        ig_j = ig_r = None
        if (self.self_policy.w_epi_j != 0.0
                or self.self_policy.w_epi_r != 0.0):
            # Action dependence: playing a_i makes the opponent's next
            # reciprocity stimulus f' = a_i, so the expected information
            # gain of observing them differs per action.
            #   f' = 1 - 2*a_i (my action); the exact IG additionally
            #   marginalises their simultaneous move (v3.7.5).
            ig_j = np.array([
                self.inversion.expected_infogain_exact(
                    1.0 - 2.0 * a, g_me, f_me)
                for a in (COOP, DEFECT)], dtype=float)
            if self.reward_model is not None:
                # v3.7.2 (path A): expected KL of the conjugate
                # posterior, in nats:
                #   I_R(a) = sum_{a_j} p_j(a_j) * 1/2 ln(1 + 1/(1+n_(a,a_j)))
                # No span normalisation needed (dimensionless), same
                # units as IG_theta.
                gain = self.reward_model.epistemic_gain()
                ig_r = np.array([
                    (p_coop_j * gain[joint_index(a, COOP)]
                     + (1.0 - p_coop_j) * gain[joint_index(a, DEFECT)])
                    for a in (COOP, DEFECT)], dtype=float)

        # --- Payoff supply by access mode ---
        if self.payoff_access == "naive":
            u_s = self.reward_model.payoff_vector("self")
            u_o = self.reward_model.payoff_vector("other")
        else:
            u_s = u_o = None

        # --- Likelihood intercept: lambda-weighted value advantage ---
        #   A_x(s) = Zbar_x(s, C) - Zbar_x(s, D)
        #   shift  = (1-gamma) * [(1-lambda)*A_self + lambda*A_other]
        #
        # [v1.9.0 — immediate/tail split retired]
        # v1.7-1.8 mixed an immediate term from R-hat (1-step) with a
        # tail term from Z. But Z already *is* the long-run value of the
        # action in this state, so the difference is the advantage, and
        # mixing sources is artificial. R-hat had been pulled in because
        # unvisited cells linger at their init value — a **learning-rate
        # problem**, not an intercept problem, and the visit-adaptive
        # learning rate of QuantileTD (lr = max(1/(1+n), lr_base))
        # removes the cause directly: one bad experience immediately
        # lowers that action's expectation. Hence the intercept is built
        # from Z alone.
        shift_i = None
        # Active from the first reward observation onward (the old
        # n_obs gates were retired in v3.7.4): with flat init the
        # pre-observation advantage is exactly 0, and the first
        # observation already carries the right sign via the
        # visit-adaptive learning rate.
        if self.qrtd is not None and self.qrtd.n_obs > 0:
            s_cur = state_index(
                COOP if self.my_last is None else int(self.my_last),
                COOP if not self.opp_actions else int(self.opp_actions[-1]))
            a_self = (self.qrtd.value(s_cur, COOP, "self")
                      - self.qrtd.value(s_cur, DEFECT, "self"))
            a_other = (self.qrtd.value(s_cur, COOP, "other")
                       - self.qrtd.value(s_cur, DEFECT, "other"))
            # **es(lambda, s) — empathy intercept computed from Z alone**
            raw_es = float(empathy_shift_z(lam, a_self, a_other))
            # Pragmatic term of -G_social: w_U * es. Since v3.8.3 the
            # former sigma_ref normalisation (a construction-time
            # constant) is folded into w_U, so w_U carries units of
            # 1/reward and the effective raw-es slope is beta_g * w_U.
            shift_i = self.w_u * raw_es
            # v3.9.6 audit channels: the three -G_social components
            # (before beta_g), logged as g_prag / g_epi_r / g_epi_j.
            self._g_prag = float(shift_i)
            self._g_epi_r = 0.0
            self._g_epi_j = 0.0
            # **Epistemic terms** (v3.1) — the EFE epistemic value is
            # added to the intercept. For a 2-action problem,
            # softmax(-G) is exactly the sigmoid of the logit
            # difference, so adding the differences preserves the EFE
            # form:
            #     logit = beta*(w_U*es + w_R*dIG_R + w_theta*dIG_j),
            #     dIG_x = IG_x(C) - IG_x(D).
            # Why it self-extinguishes: IG_R decays as visits
            # accumulate (~1/(1+n)) and IG_j shrinks as the posterior
            # contracts — exploration switches itself off with no
            # schedule.
            if self.w_ig_r > 0.0 and ig_r is not None:
                self._g_epi_r = self.w_ig_r * float(ig_r[COOP] - ig_r[DEFECT])
                shift_i += self._g_epi_r
            if self.w_ig_j > 0.0 and ig_j is not None:
                self._g_epi_j = self.w_ig_j * float(ig_j[COOP] - ig_j[DEFECT])
                shift_i += self._g_epi_j

        term = (self.qrtd.terminal_value
                if (self.qrtd is not None and self.qrtd.n_obs > 0) else None)

        if self.policy_mode == "lambda_only":
            # **Lambda-only action selection** (v2.0) — the trait axes
            # (rho, omega, eta) are retired from the focal policy.
            # State/context dependence lives in the advantages
            # A_x(s) = Zbar_x(s,C) - Zbar_x(s,D); relationship
            # dependence lives in the dynamic regulation of lambda.
            # Reciprocity-looking behaviour comes from the
            # state-dependent intercept; disposition-looking behaviour
            # from the lambda trajectory.
            # **Social EFE** (v3.4): logit = alpha + beta*(-G_social),
            #   -G_social = w_U*es + w_R*dIG_R + w_theta*dIG_j
            # (es is a C-D difference by definition, and the IG terms
            # are differences too, so this sigmoid is exactly the
            # two-action softmax(-G).)
            _z = self.alpha_bias + (0.0 if shift_i is None
                                    else self.beta_g * shift_i)
            q_c = float(1.0 / (1.0 + np.exp(-_z)))
            pol = {"ess": float("nan"), "theta_mean": {}, "pc": q_c}
        else:
            # Legacy traits path — kept for the EmpathicAgent baseline
            # (fixed-lambda comparisons); HalloRegAgent never enters here.
            pol = self.self_policy.step(theta_j, lam, f_me, g_me,
                                    p_other, p_self, ig_j=ig_j,
                                    u_self=u_s, u_other=u_o,
                                    terminal=term, shift_i=shift_i,
                                    p_coop_j=p_coop_j, ig_r=ig_r)
            q_c = self.self_policy.coop_prob_mixture(f_me, g_me, lam,
                                                     p_other, shift=shift_i)
        self._shift_used = (shift_i if shift_i is not None
                            else empathy_shift(lam, p_other))
        self._shift_oracle = empathy_shift(lam, p_other)
        action = COOP if self.rng.random() < q_c else DEFECT
        self._last_policy = pol
        self._last_qc = q_c
        # Start of the next TD transition; the state is (my last,
        # their last) **at decision time**.
        my_prev = self.my_last if self.my_last is not None else COOP
        opp_prev = self.opp_actions[-1] if self.opp_actions else COOP
        self._prev_sa = (state_index(my_prev, opp_prev), int(action))
        res = None

        # ---------- (5) Self-projection filter update ----------
        # From the opponent's viewpoint: my action is the observation,
        # their own last action is the reciprocity stimulus f, and my
        # last action is g (f/g roles exactly swapped vs the focal
        # filter).
        self_pc = np.nan
        if self.self_inversion is not None:
            self_ctx = ObservationContext(
                my_last_action=(self.opp_actions[-1] if self.opp_actions else None),
                their_last_action=self.my_last,
                round_number=len(self.my_actions))
            # [H6 projection] Record the predicted cooperation
            # probability **before** the update; measuring after would
            # leak the just-observed action into the "prediction".
            f_me = (0.0 if not self.opp_actions
                    else 1.0 - 2.0 * float(self.opp_actions[-1]))
            g_me = 1.0 - 2.0 * float(self.my_last)
            self_pc = float(self.self_inversion.predict_coop(f_me, g_me))
            self.self_inversion.update(action, self_ctx)

        # ---------- (6) State bookkeeping ----------
        self.my_last = action
        self.my_actions.append(action)
        my_rate = float(np.mean(self.my_actions))
        self.social_efe.my_coop_rate = my_rate
        self.tom.update_my_policy_belief(my_rate)
        self.inversion.my_cooperation_rate = my_rate
        if self.self_inversion is not None:
            # In the self-projection filter, the "my cooperation
            # rate" slot is filled by the opponent's rate.
            self.self_inversion.my_cooperation_rate = (
                float(np.mean([a == COOP for a in self.opp_actions]))
                if self.opp_actions else 0.5)

        # ---------- (7) Logging ----------
        self._record(action, lam, res, belief_update, reg)
        self.log["self_pred_coop"].append(float(self_pc))
        return action

    # ============================================================ Logging
    def _record(self, action: int, lam: float, res, belief_update: float,
                reg: dict) -> None:
        m = self.inversion.posterior_means()
        sd = self.inversion.posterior_stds()
        self.log["action"].append(int(action))
        self.log["lam"].append(float(lam))
        # res is unused since the trait-space policy: predicted
        # cooperation comes from the opponent filter, reliability from
        # that filter directly.
        self.log["pred_coop"].append(
            float(res.info["pc"]) if res is not None
            else float(self.inversion.predict_coop(
                0.0 if self.my_last is None else 1.0 - 2.0 * float(self.my_last),
                0.0 if not self.opp_actions
                else 1.0 - 2.0 * float(self.opp_actions[-1]))))
        self.log["reliability"].append(float(self.inversion.reliability()))
        # ---- Self-trait (rho, omega, eta) posterior ----
        sp = self.self_policy.posterior_means()
        spd = self.self_policy.posterior_stds()
        for ax in SELF_AXES:
            self.log[f"S_{ax}"].append(float(sp[ax]))
            self.log[f"SD_S_{ax}"].append(float(spd[ax]))
        self.log["policy_ess"].append(float(self._last_policy.get("ess", np.nan)))
        self.log["policy_G"].append(float(self._last_policy.get("G_mean", np.nan)))
        self.log["q_coop"].append(float(self._last_qc))
        for ax in ("alpha", "rho", "omega", "eta", "beta", "lambda_j"):
            self.log[f"E_{ax}"].append(float(m[ax]))
            self.log[f"SD_{ax}"].append(float(sd[ax]))
        self.log["belief_update"].append(float(belief_update))
        # Self-projection posterior means; NaN when the filter is
        # disabled (self_projection=False).
        pm = (self.self_inversion.posterior_means()
              if self.self_inversion is not None else None)
        for ax in ("alpha", "rho", "omega", "eta", "beta", "lambda_j"):
            self.log[f"P_{ax}"].append(
                float(pm[ax]) if pm is not None else float("nan"))


class HalloRegAgent(EmpathicAgent):
    """
    **HalloReg — hierarchical allostatic regulation agent.**

    Combines SelfModel / CoreAffect / Empathy on top of EmpathicAgent
    to regulate lambda endogenously.

    Parameters
    ----------
    w_cd : float
        Affect-context channel weight of Empathy.
    lam_gain : float
        Lambda integrator gain eta.
    lam_min, lam_max : float
        Admissible lambda interval (legacy integrator machinery).
    social_lr : float
        Learning rate of the SelfModel social baseline (setpoint drift).
    kl_scale : float
        CoreAffect arousal saturation constant kappa.
    identity_memory : bool
        Use identity-keyed memory; False treats every partner as new
        (ablation control).
    regulate : bool
        False pins lambda at its setpoint — ablation of the regulation
        mechanism itself.
    """

    LOG_KEYS = EmpathicAgent.LOG_KEYS + (
        "valence", "arousal", "lambda_aff", "lambda_ctx", "rpe", "surprise",
        "expected_reward", "baseline_reward", "pessimism", "social_distance",
        "shift_used", "shift_oracle", "qrtd_n", "value",
        "lam_sp", "allo_phi", "lam_l0", "fitness",
        "lambda_setpoint",
        # v3.9.5: Z-tilde advantages behind lambda* (A^Z, B^Z of s_t)
        "anchor_A", "anchor_B",
        # v3.9.6: -G_social components before beta_g
        "g_prag", "g_epi_r", "g_epi_j")

    def __init__(self, w_cd: float = 0.5, lam_gain: float = 0.05,
                 lam_min: float = 0.0, lam_max: float = 0.80,
                 social_lr: float = 0.02, identity_lr: float = 0.15,
                 kl_scale: float = 0.05,
                 recency_tau: float = 200.0, familiarity_scale: float = 30.0,
                 k_disc: float = 3.0,
                 lam_floor: float = 0.10, lam_ceil: float = 0.70,
                 identity_memory: bool = True, regulate: bool = True,
                 alpha_scale: float = 2.0,
                 payoff_access: str = "naive",
                 qrtd_gamma: float = 0.9, qrtd_lr: float = 0.20,
                 w_tonic: float = 0.10,
                 aff_gain: float = 0.30,
                 lam_gain_down: float = 0.45,
                 tom_es_mode: str = "mirror",
                 z_flat_init: float = None,
                 beta_g: float = 3.0, w_u: float = 40.0,
                 sp_disposition: float = 0.50,
                 bootstrap: str = "sarsa",
                 plan_sweeps: int = 1,
                 plan_update: str = "conf",
                 plan_lr: float = 0.5,
                 n_step: int = 1,
                 w_ig_r: float = 5.0,
                 w_ig_j: float = 5.0,
                 r_surv_fixed: Optional[float] = None,
                 alpha_kappa: float = 0.0,
                 anchor_b_min: float = 0.05,
                 lam_fixed: Optional[float] = None,
                 value_policy: str = "state",
                 seed_history: bool = True,
                 history_coop_mean: float = 0.55,
                 history_coop_sd: float = 0.18,
                 history_distance_slope: float = 0.0,
                 name: str = "HalloReg", **kwargs):
        # The initial lambda is set by the SelfModel setpoint; the
        # parent's lam argument is a placeholder.
        kwargs.pop("lam", None)
        super().__init__(lam=0.4, name=name, **kwargs)
        #: Fixed since v3.9.3: λ-only action selection is the sole policy
        #: path of HalloRegAgent (base class keeps "traits" for the legacy
        #: EmpathicAgent baseline).
        self.policy_mode = "lambda_only"

        self.identity_memory = bool(identity_memory)
        self.regulate = bool(regulate)

        # ---- Hierarchy ----
        self.self_model = SelfModel(
            social_lr=social_lr, identity_lr=identity_lr,
            recency_tau=recency_tau, familiarity_scale=familiarity_scale,
            k_disc=k_disc, lam_floor=lam_floor, lam_ceil=lam_ceil)
        self.core_affect = CoreAffect(self.self_model, payoffs=PAYOFF_SELF,
                                      kl_scale=kl_scale)

        # ---- Distributional value/reward learning (QRTD) ----
        # payoff_access — **default "naive"** (since v1.5.2).
        #   "naive"  : the payoff matrix is never observed directly;
        #              R-hat (1-step reward) and Z (returns) are learned
        #              from observed rewards only, and the likelihood
        #              intercept uses the learned s-hat(lambda, p). This
        #              must be the default in a non-stationary payoff
        #              environment — the old "oracle" default handed the
        #              current payoff matrix to HalloReg every round,
        #              confounding comparisons with fixed strategies.
        #   "oracle" : direct payoff observation; upper-bound reference
        #              and ablation control only.
        self.payoff_access = str(payoff_access)
        #: Social-EFE precision beta and pragmatic weight w_U (v3.4):
        #:     -G_social(C|lambda,s) = w_U*es + w_R*dIG_R + w_theta*dIG_j
        #:     P(C) = sigmoid(beta * (-G_social))
        #: Since v3.8.3 w_U absorbs the former sigma_ref constant, so
        #: the effective raw-es slope is beta_g * w_U.
        self.beta_g = float(beta_g)
        self.w_u = float(w_u)
        #: Derived beta*w_U — pragmatic-only precision; the planner's
        #: per-state q_c(s') uses it.
        self.beta_es = self.beta_g * self.w_u
        #: Fixed survival reference (None = maximin of the learned
        #: R-hat guarantee levels).
        self.r_surv_fixed = (None if r_surv_fixed is None
                             else float(r_surv_fixed))
        #: Epistemic weights of the social EFE.
        self.w_ig_r = float(w_ig_r)
        self.w_ig_j = float(w_ig_j)
        #: ToM es mode (v3.7.1): 'mirror' (default) — the opponent
        #: likelihood's es term is served by my own Z-tilde with roles
        #: swapped, so the ToM assumes the same family of utilities that
        #: actually generates my behaviour. Shared gain w_U makes the
        #: model exactly self-congruent at beta_j = beta (v3.8.3, S1):
        #:     es_j(lambda_j; s) = w_U*[(1-lambda_j)*A_self(m(s))
        #:                              + lambda_j*A_other(m(s))]
        #:     m(s) = mirror state (roles swapped to their viewpoint)
        #: With no history (f == 0 or g == 0) it falls back to the
        #: analytic form. 'analytic' is the legacy provider.
        self.tom_es_mode = str(tom_es_mode)
        if self.tom_es_mode == "mirror":
            self.inversion.es_provider = self._tom_mirror_es
        self.alpha_kappa = float(alpha_kappa)
        self.core_affect.sp_disposition = float(sp_disposition)
        # Degeneracy guard on B^Z (|B| below -> lambda* = 0.5). Z~
        # advantages live on the per-round mean-return scale and are
        # policy-attenuated (measured B^Z ~ 0.1-0.4 vs TFT), so the
        # historical R-hat guard of 0.25 (B = T - S = 5 nominally) would
        # switch the anchor off almost always. 0.05 is a provisional
        # design choice, not a fitted value.
        self.anchor_b_min = float(anchor_b_min)
        self._anchor_A = float("nan")
        self._anchor_B = float("nan")
        # Policy weight used in V(s_t) = q_c*Z~(s_t,C) + (1-q_c)*Z~(s_t,D):
        #   "state" : q_c re-evaluated at s_t with lambda_{t-1}
        #             (default since v3.9.6; option B of the E-3 review)
        #   "last"  : q_c of the previous decision, taken at s_{t-1}
        #             (v3.2-v3.9.5; kept for ablation — set the kwarg
        #             directly, e.g. HalloRegAgent(value_policy="last"))
        # Measured v3.9.6 (800R, 8 seeds, 7 dyad scenarios + 3 H1A
        # switches): no difference beyond seed noise — at beta_eff=120
        # both q_c are saturated except in the 1-2 rounds around a
        # relationship transition. The early lambda spike is unchanged
        # (it stems from the R-hat/Z-tilde initialisation scale
        # mismatch in phi, not from the q_c lag).
        if value_policy not in ("last", "state"):
            raise ValueError(f"value_policy must be 'last' or 'state'")
        self.value_policy = str(value_policy)
        self._qc_value = float("nan")
        # Counterfactual: hold lambda at a constant (regulation off,
        # lambda written directly, no Empathy clipping). None = regulated.
        self.lam_fixed = None if lam_fixed is None else float(lam_fixed)
        if self.lam_fixed is not None:
            self.regulate = False
        self.reward_model = RewardModel(lr=max(qrtd_lr * 2, 0.02))
        self.qrtd = QuantileTD(gamma=qrtd_gamma, lr=qrtd_lr,
                               seed=(kwargs.get("seed", 0) or 0) + 5171)
        self.reward_model.set_scale(float(PAYOFF_SELF.max()
                                          - PAYOFF_SELF.min()))
        # v2.7.0: Z-tilde = (1-gamma)Z, so the Huber threshold is set
        # on the **reward scale**.
        self.qrtd.set_scale(float(PAYOFF_SELF.max() - PAYOFF_SELF.min()))
        self.qrtd.bootstrap = str(bootstrap)
        #: Model-based planning sweeps per round (0 = pure TD).
        self.plan_sweeps = int(plan_sweeps)
        self.plan_update = str(plan_update)
        self.plan_lr = float(plan_lr)
        #: n of on-policy n-step SARSA (1 = plain 1-step).
        self.n_step = max(1, int(n_step))

        self._prev_sa = None            # previous (state, action) for TD
        self._state_visits = np.zeros(4)   # empirical state occupancy d(s)

        # ---- Pre-experiment social history ----
        # Individuals do not enter society blank: preloading a network
        # of relationships (close / acquaintance / distant) makes the
        # distance-weighted setpoint actually operative. Without it,
        # every agent's lambda_0 collapses to the same 0.40 and social
        # distance has no effect on lambda.
        if seed_history:
            self.self_model.seed_social_history(
                PAYOFF_SELF, gamma=qrtd_gamma,
                coop_mean=history_coop_mean,
                coop_sd=history_coop_sd,
                distance_coop_slope=history_distance_slope,
                rng=np.random.default_rng(
                    (kwargs.get("seed", 0) or 0) * 7919 + 104729))

        # Flat Z-tilde initialisation (v3.7.3): every Z(s,a) starts at
        # the payoff-support midpoint c0 = (r_min + r_max)/2 (2.5 in the
        # standard PD) — the minimal-arbitrariness location on a bounded
        # support, not a magic number. Empirical basis (800R): c0 = 2.5
        # gives HR-HR CC 0.893 +/- 0.017 with low variance, ALLD 0.039,
        # best mixed payoff; c0 is genuinely sensitive (c0 = 1.0
        # collapses HR-HR to 0.287). An early lambda spike (<= 1.0,
        # gone by ~R20) reappears from initial optimism — outside the
        # evaluation window; recorded honestly.
        # -------- Lambda-mapping constants (sole path since v3.9.3) -----
        # IG_theta: exact 1-step-ahead (double marginalisation over the
        #   opponent's simultaneous move) — the frozen approximation was
        #   removed in v3.9.3 along with its mode switch.
        # Lambda mapping (eslin): lam = clip(lam_star + phi - 1/2, 0, 1).
        #   Anchor lam_star is the analytic *switch* point derived from the
        #   learned reward structure R-hat (model source):
        #     lam0 = clip(-dU_own / den, 0, 1),  den = dU_par - dU_own,
        #     dU_x(p) = p*[Rx(CC)-Rx(DC)] + (1-p)*[Rx(CD)-Rx(DD)],
        #     lam_star = lam0 + ln2 / (beta_g * w_u * den).
        #   Guard: |den| < 0.25 (undifferentiated early estimates) falls
        #   back to lam_star = 0.5, i.e. the neutral linear map lam = phi.
        # Z-tilde flat initialisation: support midpoint c0 with a small
        #   tau-fan spread (v3.7.3); the social-prior init was removed.
        if z_flat_init is not None:
            _c0 = float(z_flat_init)
        else:
            _po = self.core_affect.payoffs
            _c0 = 0.5 * (float(np.min(_po)) + float(np.max(_po)))
        self.qrtd.reinit(_c0, 0.5)

        # v2.2: lambda starts neutral (0.5); social-history information is
        # carried by the additive bias alpha, not by the lambda init.
        lam0 = 0.5
        self.empathy = Empathy(lam_init=lam0, w_cd=w_cd, gain=lam_gain,
                               w_tonic=w_tonic, aff_gain=aff_gain, gain_down=lam_gain_down,
                               lam_min=lam_min, lam_max=lam_max,
                               alpha_scale=alpha_scale)

        #: Social-history cooperation bias alpha — fixed once after
        #: history seeding; an additive logit term, alpha ~ N(0, kappa^2):
        #: "what kind of social world do I come from".
        self.alpha_bias = self.self_model.cooperation_bias(
            kappa=self.alpha_kappa)
        # Target intercept I0 of the compensatory lambda setpoint
        # (v2.2). With logit = alpha + beta_es*es(lambda, s), neutral
        # behaviour (logit 0) requires es* = -alpha/beta_es; the
        # setpoint must aim at this value to stay consistent with
        # alpha.
        self.core_affect.sp_i0 = float(-self.alpha_bias / max(self.beta_es,
                                                              1e-6))

        self.lam = lam0
        self.social_efe.lam = lam0
        self._identity: Optional[int] = None

    # ============================================================ Partner switch
    def begin_partner(self, identity: int) -> None:
        """
        Partner-identity notification from the environment.

        - notify SelfModel of the identity observation,
        - inject that identity's theta prior into OpponentInversion
          (re-encounters resume from the remembered relationship),
        - inject the expected-reward prior into CoreAffect,
        - reset lambda to the **currently updated setpoint**.

        identity_memory=False treats every partner as new (ablation).
        """
        pid = int(identity) if self.identity_memory else None
        self._identity = pid
        if pid is not None:
            self.self_model.observe_identity(pid)
        self.inversion.set_prior(self.self_model.theta_prior(pid), reinit=True)
        self.core_affect.begin_partner(pid)
        lam0 = self.self_model.lambda_setpoint(self.core_affect.payoffs)
        self.lam = self.empathy.reset(lam0)
        self.social_efe.lam = self.lam

    def current_lambda(self) -> float:
        return self.lam

    # ============================================================ Lambda regulation
    def _regulate(self, observed_state: int, opp_action: int,
                  inferred: dict) -> dict:
        """
        One round of hierarchical lambda regulation.

        1. Non-stationary payoff alignment — pass the current payoff
           vector to CoreAffect.
        2. CoreAffect.step(observation) -> valence, arousal, lambda_aff
           (updates the per-identity reward distribution and commits it
           to SelfModel).
        3. Commit the OpponentInversion posterior (theta, dist) to
           SelfModel.
        4. Allostatic surplus phi and the model-derived switch anchor
           lambda_star produce lambda via the eslin map.
        """
        # --- 1. Current payoffs (non-stationary schedules) ---
        self.core_affect.set_payoffs(PAYOFF_SELF)

        # --- 1b. QRTD learning ---
        # The observed joint outcome updates (i) the 1-step reward
        # model R-hat and (ii) the return values Z. Z needs a
        # transition from the previous (state, action), so it only
        # updates once _prev_sa exists.
        from AIF_IPD.core.constants import PAYOFF_OTHER
        j = int(observed_state)
        r_s, r_o = float(PAYOFF_SELF[j]), float(PAYOFF_OTHER[j])
        self.reward_model.update(j, r_s, r_o)
        v_now = v_shift = None
        v_vec = None
        if self._prev_sa is not None:
            s_prev, a_prev = self._prev_sa
            my_prev = self.my_actions[-1] if self.my_actions else COOP
            s_now = state_index(my_prev, int(opp_action))
            # Comparing Z(s,a) before/after the update yields the
            # **shift in situational-value beliefs** — the arousal
            # driver (a collapse of prospects, not a reward shift).
            # **True SARSA — hold each transition for one round**
            # (v2.6.0): the transition completed at round t is
            # (s_{t-1}, a_{t-1}) -> s_t, but its bootstrap needs
            # a' = a_t which is not chosen yet; the transition held at
            # t-1, (s_{t-2}, a_{t-2}) -> s_{t-1}, has its a' = a_{t-1}
            # **already observed** — apply that one now.
            if self._pending is None:
                self._pending = []
            if not self._pending:
                before = after = self.qrtd.z_self.values[s_prev, a_prev].copy()
            elif self.n_step <= 1:
                p_s, p_a, p_rs, p_ro, p_sn = self._pending[0]
                before = self.qrtd.z_self.values[p_s, p_a].copy()
                self.qrtd.update(p_s, p_a, p_rs, p_ro, p_sn, a_prev)
                after = self.qrtd.z_self.values[p_s, p_a]
                self._pending = []
            else:
                # **On-policy n-step SARSA** — complete the oldest
                # transition with the discounted sum of intermediate
                # rewards and a gamma^n bootstrap:
                #   y = sum_{k<n} gamma^k (1-gamma) r_{t0+k}
                #       + gamma^n * Z(s_{t0+n}, a_{t0+n})
                # The bootstrap weight shrinks gamma -> gamma^n, so
                # optimism injected by frozen cells decays
                # geometrically (Sutton & Barto ch. 7).
                self.qrtd.note_reward(r_s, r_o)   # r-bar for the n-step path
                before = self.qrtd.z_self.values[
                    self._pending[0][0], self._pending[0][1]].copy()
                after = before
                if len(self._pending) >= self.n_step:
                    p_s, p_a = self._pending[0][0], self._pending[0][1]
                    g_s = g_o = 0.0
                    _wr = 1.0 - self.qrtd.gamma
                    for _k, _e in enumerate(self._pending):
                        g_s += (self.qrtd.gamma ** _k) * _wr * _e[2]
                        g_o += (self.qrtd.gamma ** _k) * _wr * _e[3]
                    _gn = self.qrtd.gamma ** len(self._pending)
                    _bs = self.qrtd.bvec(s_prev, a_prev, "self")
                    _bo = self.qrtd.bvec(s_prev, a_prev, "other")
                    self.qrtd.apply_target(p_s, p_a,
                                           g_s + _gn * _bs,
                                           g_o + _gn * _bo)
                    after = self.qrtd.z_self.values[p_s, p_a]
                    self._pending.pop(0)
            self._pending.append((s_prev, a_prev, r_s, r_o, s_now))

            # --- (A) Model-based planning sweep ---
            if self.plan_sweeps > 0 and self.reward_model.n_obs > 3:
                # Per-state p_j — state s = (my last, their last) is
                # exactly the ToM context (f, g), so conditional
                # opponent strategies (TFT etc.) survive inside the
                # model.
                _pj = np.array([
                    float(self.inversion.predict_coop(
                        1.0 - 2.0 * float(ss // 2), 1.0 - 2.0 * float(ss % 2)))
                    for ss in range(4)])
                # Continuous policy per state:
                #   q_c(s') = sigmoid(beta_es * es(lambda_t, s')).
                # (Planning uses the pragmatic term only; the epistemic
                # terms self-extinguish later anyway.)
                if self.qrtd.n_obs > 0:
                    _qc = np.empty(4)
                    for _s in range(4):
                        _as = (self.qrtd.value(_s, COOP, "self")
                               - self.qrtd.value(_s, DEFECT, "self"))
                        _ao = (self.qrtd.value(_s, COOP, "other")
                               - self.qrtd.value(_s, DEFECT, "other"))
                        _es = (1.0 - self.lam) * _as + self.lam * _ao
                        _qc[_s] = 1.0 / (1.0 + np.exp(-self.beta_es * _es))
                else:
                    _qc = np.full(4, float(self._last_qc))
                self.qrtd.plan_sweep(
                    self.reward_model,
                    p_coop_j=_pj,
                    p_coop_self=_qc,
                    n_sweeps=self.plan_sweeps,
                    update=self.plan_update, lr=self.plan_lr)
            span_v = float(self.core_affect.payoffs.max()
                           - self.core_affect.payoffs.min())
            span_v = max(span_v, 1e-6)   # v2.7.0: Z-tilde is on reward scale
            v_shift = float(np.mean(np.abs(after - before)) / span_v)
            # Valence material: the full Z quantile vector of the
            # just-experienced (state, action). SelfModel keeps an EMA
            # of it as "the expected-reward distribution of this
            # partner", and its median rank against the population
            # reference becomes valence.
            # **Marginalise under the policy** (v1.7.0): using the raw
            # cell of the step just taken lets visit luck oscillate the
            # sign; valence should ask "is this relationship good", so
            #     Q = sum_s d(s) * sum_a pi(a|s) * Z(s,a)
            # with d(s) the empirical state occupancy and pi the
            # current cooperation probability.
            self._state_visits[s_prev] += 1.0
            d = self._state_visits / max(self._state_visits.sum(), 1e-9)
            pc = float(self._last_qc)
            v_vec = np.zeros_like(after)
            for ss in range(4):
                if d[ss] <= 1e-12:
                    continue
                # v3.7.2: read the behaviour-effective quantile vector
                # (bvec), keeping the valence material consistent with
                # the values behaviour actually uses.
                v_vec += d[ss] * (pc * self.qrtd.bvec(ss, 0, "self")
                                  + (1.0 - pc) * self.qrtd.bvec(ss, 1, "self"))
            v_now = float(np.median(v_vec))
            # **Between-state fitness** — is my current state good
            # within this relationship:
            #   V(s') = pi(C)*Z(s',C) + (1-pi)*Z(s',D)
            #   valence = 2 * sum_{s': V(s')<=V(s)} d(s') - 1
            # An occupancy-weighted percentile, so rarely visited
            # states are not overvalued. Rank statistics use the **raw
            # learned values** (percentiles are invariant to common
            # shifts): behaviour uses shrinkage, ranks use raw.
            _zs = self.qrtd.z_self.values
            _vs = np.array([pc * float(_zs[ss, COOP].mean())
                            + (1.0 - pc) * float(_zs[ss, DEFECT].mean())
                            for ss in range(4)], dtype=float)
            # **Exclude the current state from the reference.**
            # Including it puts its own mass into the percentile, so
            # the most-occupied state ranks high merely by being
            # occupied (measured pathology: vs ALLD, growing DD
            # occupancy pushed lambda to 0.79 — 0.94 cooperation with
            # an exploiter). Same reason the current partner is
            # excluded in between-relationship comparisons.
            _mask = np.ones(4, dtype=bool)
            _mask[s_prev] = False
            _den = float(d[_mask].sum())
            if _den > 1e-9:
                _num = float(d[_mask & (_vs <= _vs[s_prev])].sum())
                _st_val = float(2.0 * (_num / _den) - 1.0)
            else:
                _st_val = 0.0

        # --- 2. Core affect ---
        # Pass whether the opponent cooperated — SelfModel accumulates
        # the distance-weighted cooperation rate (the source of the
        # prosociality setpoint).
        affect = self.core_affect.step(
            observed_state, opponent_cooperated=(int(opp_action) == COOP),
            value_vector=v_vec, value_shift=v_shift,
            state_valence=(_st_val if self._prev_sa is not None else None),
            adv=(self._occ_adv() if self.policy_mode == "lambda_only"
                 else None),
            z_snapshot=(self.qrtd.z_self.values
                        if self.qrtd is not None else None))

        # --- 3. Commit partner memory ---
        # The self-trait posterior is stored too, so re-encounters
        # restore "this is the stance I used with this person".
        self.self_model.commit_self_theta(
            self._identity, self.self_policy.posterior_means(),
            self.self_policy.posterior_stds())
        self.self_model.commit_theta(self._identity, inferred,
                                     self.inversion.posterior_stds())

        # --- 4. Lambda update: allostatic direct mapping ---
        #     E_t    = expected per-round intake of this relationship
        #     r_surv = max_a min_{a_j} R-hat(a, a_j)  (guarantee level)
        #     top    = R-hat(C, C)  (mutual-cooperation optimum)
        #     phi    = (E_t - r_surv) / (top - r_surv)  (surplus ratio)
        #     lam    = clip(lam_star + clip(phi,0,1) - 1/2, 0, 1)
        # Intake below survival drives lambda -> 0 (pre-emptive
        # self-protection, the allostatic-deviation response); surplus
        # raises it. r_surv and top come from the learned R-hat, so a
        # payoff-regime change recomputes the reference points — the
        # direct mechanism of non-stationary adaptation.
        if self.regulate:
            # Uninformed default (n_obs <= 3): historical affine midpoint
            # 0.25, kept verbatim for behaviour preservation.
            lam_allo = 0.25
            self._allo_phi = np.nan
            self._lam_l0 = np.nan
            if self.reward_model.n_obs > 3:
                # Early fallback for E_t: a **ToM-based 1-step
                # prediction** (not Z-tilde),
                #   E_t = q_c*[p_j*R(CC) + (1-p_j)*R(CD)]
                #       + (1-q_c)*[p_j*R(DC) + (1-p_j)*R(DD)],
                # because occupancy-weighted Z would keep unvisited
                # cooperative branches at the prior and contaminate E
                # optimistically (measured: lambda up to 0.65 vs ALLD).
                # The ToM weight p_j erases unvisited branches
                # (exploiter -> p_j -> 0) and reacts **pre-emptively**
                # at the opponent model's update speed — predictive
                # (allostatic), not reactive (homeostatic) regulation.
                qc = float(self._last_qc)
                rv = self.reward_model.payoff_vector("self")
                if (self.value_policy == "state" and self.qrtd is not None
                        and self.qrtd.n_obs > 3):
                    # v3.9.6 (default): re-evaluate the cooperation
                    # probability **at s_t** with the previous lambda
                    # (pragmatic term only), instead of carrying the
                    # previous round's q_c computed at s_{t-1}. Using
                    # lambda_{t-1} breaks the q_c -> lambda -> phi ->
                    # V -> q_c loop without a state mismatch.
                    _as = (self.qrtd.value(observed_state, COOP, "self")
                           - self.qrtd.value(observed_state, DEFECT, "self"))
                    _ao = (self.qrtd.value(observed_state, COOP, "other")
                           - self.qrtd.value(observed_state, DEFECT, "other"))
                    _z = (self.alpha_bias + self.beta_g * self.w_u
                          * float(empathy_shift_z(self.lam, _as, _ao)))
                    qc = float(1.0 / (1.0 + np.exp(-_z)))
                self._qc_value = qc
                if (self.qrtd is not None
                        and self.qrtd.n_obs > 3):
                    # **Z-tilde based E_t** (v3.2) — long-run value
                    # drives lambda.
                    #   E_t = q_c·Z̃(s_t, C) + (1−q_c)·Z̃(s_t, D)
                    # Z-tilde is on the per-round mean-return scale,
                    # directly comparable to r_surv. Conditioning on
                    # the current state makes lambda **state
                    # dependent** — it drops in bad phases even within
                    # a good relationship. (The old occupancy-weighted
                    # failure predates the planning sweep; now all 8
                    # cells are backed up by R-hat and p_j(s) every
                    # round, removing that contamination source.)
                    e_t = (qc * self.qrtd.value(observed_state, COOP, "self")
                           + (1 - qc) * self.qrtd.value(observed_state,
                                                        DEFECT, "self"))
                else:
                    _f = 1.0 - 2.0 * float(self.my_last
                                           if self.my_last is not None else 0)
                    _g = (1.0 - 2.0 * float(self.opp_actions[-1])
                          if self.opp_actions else 0.0)
                    pj = float(self.inversion.predict_coop(_f, _g))
                    e_t = (qc * (pj * rv[0] + (1 - pj) * rv[1])
                           + (1 - qc) * (pj * rv[2] + (1 - pj) * rv[3]))
                r_surv = (self.r_surv_fixed
                          if self.r_surv_fixed is not None else
                          float(max(min(rv[0], rv[1]), min(rv[2], rv[3]))))
                top = float(rv[0])
                if top - r_surv > 1e-6:
                    phi_a = (e_t - r_surv) / (top - r_surv)
                    self._allo_phi = float(phi_a)
                    _ph = float(np.clip(phi_a, 0.0, 1.0))
                    # **Analytic switch point from the generalised
                    # empathy shift (v3.9.4; arch. doc §3.3.4)**
                    #   es^Z(lambda, s) = A^Z(s) + lambda * B^Z(s)
                    #   A^Z(s) = Z~_self(s_t, C) - Z~_self(s_t, D)
                    #   B^Z(s) = [Z~_other(s_t, C) - Z~_other(s_t, D)] - A^Z
                    #   lambda*(s) = (ln2/beta_eff - A^Z) / B^Z,
                    #   beta_eff = beta_g * w_u
                    # These are the same quantities that build the action
                    # logit, so lambda* is exactly the lambda at which the
                    # pragmatic term alone yields q_c = 2/3.
                    # [R-hat anchor retired in v3.9.4] The former one-step
                    # anchor A = E_{a_j}[R_s(C,a_j) - R_s(D,a_j)] lived on a
                    # different valuation than the logit (R-hat + ToM p_j
                    # vs Z~), so "switch point" was not a property of the
                    # policy actually used. Measured consequence of the
                    # change (800R, 5 opponents, noise .05): lambda level
                    # drops by 0.2-0.3 vs reciprocators (A^Z >= 0 under an
                    # established cooperative policy -> lambda* clips to
                    # ~0), behaviour (CC, DD, payoff, recovery time)
                    # unchanged within seed noise.
                    if self.qrtd is not None and self.qrtd.n_obs > 3:
                        _do = float(self.qrtd.value(observed_state, COOP, "self")
                                    - self.qrtd.value(observed_state, DEFECT, "self"))
                        _dp = float(self.qrtd.value(observed_state, COOP, "other")
                                    - self.qrtd.value(observed_state, DEFECT, "other"))
                    else:
                        _do, _dp = 0.0, 0.0        # -> degenerate guard below
                    _den = _dp - _do
                    self._anchor_A = float(_do)
                    self._anchor_B = float(_den)
                    if abs(_den) > self.anchor_b_min:
                        _l0 = -_do / _den + (np.log(2.0)
                                             / (self.beta_g * self.w_u
                                                * _den))
                        _l0 = float(np.clip(_l0, 0.0, 1.0))
                    else:
                        _l0 = 0.5
                    self._lam_l0 = _l0
                    lam_allo = float(np.clip(_l0 + _ph - 0.5, 0.0, 1.0))
            self.lam = lam_allo
            self.empathy.lam = self.lam
        else:
            # regulation off (ablation)
            self.lam = (self.lam_fixed if self.lam_fixed is not None
                        else self.empathy.lam)
            self.empathy.lam = self.lam
        self.social_efe.lam = self.lam

        return {"valence": affect["valence"], "arousal": affect["arousal"],
                "lambda_aff": affect["lambda_aff"], "lambda_ctx": 0.0,
                "lam_sp": affect.get("lam_sp", float("nan")),
                "fitness": affect.get("fitness", float("nan")),
                "allo_phi": float(getattr(self, "_allo_phi", float("nan"))),
                "lam_l0": float(getattr(self, "_lam_l0", float("nan"))),
                "anchor_A": float(getattr(self, "_anchor_A", float("nan"))),
                "anchor_B": float(getattr(self, "_anchor_B", float("nan"))),
                "rpe": affect["rpe"], "surprise": affect["surprise"],
                "expected_reward": self.core_affect.expected_reward(),
                "baseline_reward": affect["r_base"],
                "pessimism": affect["pessimism"],
                # Both the intercept actually used and the oracle
                # analytic value are logged, so naive-mode learning can
                # be audited post hoc.
                "shift_used": getattr(self, "_shift_used", float("nan")),
                "g_prag": getattr(self, "_g_prag", float("nan")),
                "g_epi_r": getattr(self, "_g_epi_r", float("nan")),
                "g_epi_j": getattr(self, "_g_epi_j", float("nan")),
                "shift_oracle": getattr(self, "_shift_oracle", float("nan")),
                "value": affect["value"],
                "qrtd_n": float(self.qrtd.n_obs
                                if self.qrtd is not None else 0),
                "social_distance": self.self_model.social_distance(
                    self._identity),
                "lambda_setpoint": self.self_model.lambda_setpoint()}

    # ============================================================ Logging
    def _record(self, action, lam, res, belief_update, reg) -> None:
        super()._record(action, lam, res, belief_update, reg)
        # First round has no observation, so reg is empty — fill
        # with neutral values.
        defaults = {"valence": 0.0, "arousal": 0.0, "lambda_aff": 0.0,
                    "lam_sp": float("nan"), "fitness": float("nan"),
                    "allo_phi": float("nan"),
                    "lam_l0": float("nan"),
                    "anchor_A": float("nan"), "anchor_B": float("nan"),
                    "lambda_ctx": 0.0, "rpe": 0.0, "surprise": 0.0,
                    "expected_reward": self.core_affect.expected_reward(),
                    "baseline_reward":
                        self.self_model.reference_median(
                            exclude=self._identity),
                    "pessimism": 0.0,
                    "social_distance": self.self_model.social_distance(
                        self._identity),
                    "lambda_setpoint": self.self_model.lambda_setpoint(),
                    "shift_used": getattr(self, "_shift_used", float("nan")),
                "g_prag": getattr(self, "_g_prag", float("nan")),
                "g_epi_r": getattr(self, "_g_epi_r", float("nan")),
                "g_epi_j": getattr(self, "_g_epi_j", float("nan")),
                    "shift_oracle": getattr(self, "_shift_oracle",
                                            float("nan")),
                    "value": float("nan"),
                    "qrtd_n": float(self.qrtd.n_obs
                                    if self.qrtd is not None else 0)}
        for k, v in defaults.items():
            self.log[k].append(float(reg.get(k, v)))
