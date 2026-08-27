"""
experiments.h6_recovery
=======================

**H6 — are parameter recovery and projection robust?**

Two logically independent subtasks.

Task A — parameter recovery
---------------------------
The standard check of recovery validity: can the model invert the
data-generating process it assumes?

  1. Draw a true theta* = (alpha, rho, omega, eta, beta, lambda_j)
     from the interior of the prior support.
  2. `LikelihoodAgent(theta*)` emits actions directly from the
     inference's generative likelihood.
  3. The focal is a `ProbeAgent` — cooperating with probability 0.5
     independently each round.
  4. Apply `OpponentInversion` to the observation sequence and
     compare theta-hat with theta*.

[Why a ProbeAgent — identifiability]
Recovery depends on whether the design matrix (1, f, g, f*g) is
actually spanned. A policy-driven focal (e.g. always cooperating)
makes f constant, so alpha and rho become collinear and rho, omega,
eta are unidentifiable. A random probe balances f across both levels
and fills all four cells of its combination with the opponent's own
history g.

[Predicted in advance — a limitation to state explicitly]
lambda_j enters the likelihood only as an **intercept**,
s(lambda_j, p) = (T-S)*lambda_j + ... . Under fixed payoffs (T-S) = 5
is constant, so the lambda_j term is perfectly collinear with alpha
and the two axes are **not separately identifiable**. Two conditions
therefore run side by side:

  - condition fixed  : the standard PD. lambda_j recovery is
                       **predicted in advance to fail**.
  - condition varied : (T, S) is modulated per block to centre the
                       lambda_j regressor (T-S), predicted to break
                       the collinearity and make lambda_j
                       identifiable.

This contrast diagnoses "recovery failure" not as a defect but as a
**structural consequence of identifiability**; if the diagnosis is
right, recovery must return in the varied condition.

  - Confirmatory A1 : in the fixed condition, corr(theta-hat, theta*)
                      > 0 on each of the five axes (alpha, rho,
                      omega, eta, beta).
  - Confirmatory A2 : lambda_j correlation in varied > in fixed.
  - Exploratory     : per-axis bias and RMSE, convergence over
                      rounds.

[An additional advance prediction about beta]
beta multiplies the **entire** linear predictor, so it changes only
the extremity of choice probabilities, never the sign or ordering of
the linear terms. Hence beta is (i) multiplicatively entangled with
alpha, rho, omega and eta (directions exist in which only beta*alpha
is identified), and (ii) at moderate beta the sigmoid is already near
saturation, so the observation likelihood is insensitive to it. **The
recovery correlation for beta is therefore predicted in advance to be
lower than for the other axes.** This is a known identification limit
of logistic choice models, not a particle-filter defect, and if it
comes out low as predicted it is reported as such — no forced
"supported" verdict.

Task B — self-projection
------------------------
HalloReg's depth-2 perspective taking constructs "how does the other
see me" with a self-projection filter theta-hat_self (the same
inversion applied to one's own action history). For that projection
to be valid it must **agree with what an actual external observer
infers about me**.

  Primary index — **projection fidelity**
      HalloReg A plays HalloReg B.
        - A's theta-hat_self = what A thinks "B sees of me"
        - B's theta-hat      = what B actually inferred about A
      The two vectors are compared per axis; r > 0 means the
      projection is veridical. This is the validity condition of the
      depth-2 recursion, testing directly whether the projection
      **approximates the other's viewpoint** rather than being a
      self-flattering fantasy.

  Secondary index — predictive calibration
      AUC and Brier skill score of the cooperation probability the
      self-projection filter emits each round (before updating)
      against the focal's actual behaviour.

      **Advance prediction (important).** The focal's round-level
      actions are sampled from softmax(-G_social), so much of their
      variability is **policy sampling noise**, which is
      unpredictable in principle; the AUC is therefore **predicted in
      advance** to sit near 0.5. That is not a failure of projection
      but a question of what is predictable at all, so this index is
      registered as **exploratory**, not confirmatory.

  - Confirmatory B1 : projection fidelity — corr(theta-hat_self,
                      theta-hat_partner) > 0 on alpha, rho, beta and
                      lambda_j.
  - Exploratory     : fidelity on omega and eta, AUC, Brier skill
                      score, calibration by opponent type.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from AIF_IPD.core.constants import (
    COOP, DEFECT, reset_payoffs, set_payoff_matrix,
)
from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import ALL_TYPES, LikelihoodAgent, ProbeAgent, TYPE_LABEL_KO
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, corr_boot, one_sample_perm, perm_test,
)
from AIF_IPD.ipd.population import type_spec
from AIF_IPD.ipd.sim import run_many
from AIF_IPD.ipd.tom.inversion import (
    ObservationContext, OpponentInversion, THETA_AXES,
)
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json

LOGGER = get_logger("HalloReg.H6")

#: Range from which true theta is drawn (inside the prior support,
#: avoiding boundary-clipping artefacts)
TRUE_RANGE = {
    "alpha": (-2.0, 2.0), "rho": (-1.5, 2.5), "omega": (-1.5, 1.5),
    "eta": (-1.5, 1.5), "beta": (1.0, 6.0), "lambda_j": (0.05, 0.95),
}

#: Per-block (R, T, S, P) in the varied condition. The lambda_j
#: regressor is (T - S), so this value changes greatly across blocks
#: and averages near zero, breaking collinearity with the intercept
#: alpha. The third block is a non-PD game with T < S — a deliberate
#: design choice for identification, with no effect outside this
#: experiment.
VARIED_BLOCKS = [
    (3.0, 5.0, 0.0, 1.0),      # T-S = +5  (standard PD)
    (3.0, 2.0, 2.0, 1.0),      # T-S =  0  (regressor cancelled)
    (3.0, 0.0, 5.0, 1.0),      # T-S = -5  (sign inverted)
    (3.0, 4.0, 1.0, 1.0),      # T-S = +3
]

AXIS_LABEL_KO = {"alpha": "alpha (cooperation bias)",
                 "rho": "rho (reciprocity)",
                 "omega": "omega (inertia)",
                 "eta": "eta (outcome contingency)",
                 "beta": "beta (precision)",
                 "lambda_j": "lambda_j (their empathy)"}
IDENTIFIED_AXES = ("alpha", "rho", "omega", "eta", "beta")


# ================================================================ Task A
def _recover_one(theta_true: Dict[str, float], n_rounds: int, seed: int,
                 varied: bool, n_particles: int) -> Tuple[Dict[str, float], np.ndarray]:
    """
    One recovery trial; returns (final theta-hat, per-round theta-hat
    trajectory (T, D)).

    The simulation loop is run directly: the focal is a random probe
    rather than an active-inference agent, so run_dyad's AIF path is
    unnecessary and only the inference needs to see the observation
    sequence.
    """
    rng_gen = LikelihoodAgent(**theta_true, seed=seed)
    probe = ProbeAgent(0.5, seed=seed + 1)
    inv = OpponentInversion(n_particles=n_particles, seed=seed + 2)

    my_actions: List[int] = []
    opp_actions: List[int] = []
    traj = np.zeros((n_rounds, len(THETA_AXES)))

    for t in range(n_rounds):
        # ---- varied condition: change payoffs per block to modulate
        # the lambda_j regressor ----
        if varied:
            b = min(len(VARIED_BLOCKS) - 1,
                    int(t * len(VARIED_BLOCKS) / max(n_rounds, 1)))
            set_payoff_matrix(*VARIED_BLOCKS[b])

        # ---- Action emission ----
        a_j = rng_gen.act()          # opponent (generating process)
        a_i = probe.act()            # focal (random probe)
        rng_gen.observe(a_i)

        # ---- Inference update (tense: f = my_{t-1},
        # g = opp_{t-1}) ----
        ctx = ObservationContext(
            my_last_action=(my_actions[-1] if my_actions else None),
            their_last_action=(opp_actions[-1] if opp_actions else None),
            round_number=t)
        inv.my_cooperation_rate = (float(np.mean([a == COOP for a in my_actions]))
                                   if my_actions else 0.5)
        inv.update(a_j, ctx)

        my_actions.append(a_i)
        opp_actions.append(a_j)
        m = inv.posterior_means()
        traj[t] = [m[ax] for ax in THETA_AXES]

    if varied:
        reset_payoffs()
    return inv.posterior_means(), traj


def run_recovery(cfg: Config, reg: Registry) -> dict:
    """Task A — parameter recovery (fixed vs varied conditions)."""
    n_sim = cfg.seeds
    rng = np.random.default_rng(41)
    truths = []
    for i in range(n_sim):
        truths.append({ax: float(rng.uniform(*TRUE_RANGE[ax]))
                       for ax in THETA_AXES})

    out = {}
    for cond, varied in (("fixed", False), ("varied", True)):
        LOGGER.info("  [H6-A] recovery condition '%s' — %d trials x "
                    "%d rounds",
                    cond, n_sim, cfg.rounds)
        T = np.zeros((n_sim, len(THETA_AXES)))     # true
        H = np.zeros((n_sim, len(THETA_AXES)))     # estimated
        trajs = np.zeros((n_sim, cfg.rounds, len(THETA_AXES)))
        for i, th in enumerate(truths):
            est, tr = _recover_one(th, cfg.rounds, seed=20_000 + i * 13,
                                   varied=varied, n_particles=cfg.n_particles)
            T[i] = [th[ax] for ax in THETA_AXES]
            H[i] = [est[ax] for ax in THETA_AXES]
            trajs[i] = tr
        out[cond] = {"true": T, "hat": H, "traj": trajs}

    # ---- Confirmatory A1: five-axis recovery in the fixed
    # condition ----
    corrs_fixed = {}
    for d, ax in enumerate(THETA_AXES):
        c = corr_boot(out["fixed"]["true"][:, d], out["fixed"]["hat"][:, d],
                      seed=d)
        corrs_fixed[ax] = c
        if ax in IDENTIFIED_AXES:
            reg.confirm("H6", f"recovery (fixed) {AXIS_LABEL_KO[ax]}: "
                        f"r > 0",
                        c["p"], direction_ok=bool(c["r"] > 0),
                        effect=f"r={c['r']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]")
        else:
            reg.explore("H6", f"recovery (fixed) {AXIS_LABEL_KO[ax]} "
                              f"(collinear with alpha — failure "
                              f"predicted)", c["p"],
                        effect=f"r={c['r']:.3f}")

    # ---- Confirmatory A2: improved lambda_j recovery in varied ----
    corrs_varied = {}
    for d, ax in enumerate(THETA_AXES):
        corrs_varied[ax] = corr_boot(out["varied"]["true"][:, d],
                                     out["varied"]["hat"][:, d], seed=100 + d)
    d_lj = THETA_AXES.index("lambda_j")
    # Bootstrap test on the difference of correlations (paired, since
    # both conditions share the same set of true values)
    rng2 = np.random.default_rng(77)
    n_boot = 3000
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng2.integers(0, n_sim, n_sim)
        tf, hf = out["fixed"]["true"][i, d_lj], out["fixed"]["hat"][i, d_lj]
        tv, hv = out["varied"]["true"][i, d_lj], out["varied"]["hat"][i, d_lj]
        rf = np.corrcoef(tf, hf)[0, 1] if tf.std() > 0 and hf.std() > 0 else 0.0
        rv = np.corrcoef(tv, hv)[0, 1] if tv.std() > 0 and hv.std() > 0 else 0.0
        diffs[b] = rv - rf
    p_diff = (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    obs_diff = corrs_varied["lambda_j"]["r"] - corrs_fixed["lambda_j"]["r"]
    reg.confirm("H6", "lambda_j recovery: varied (payoff modulation) "
                "> fixed — identifiability diagnosis",
                p_diff, direction_ok=bool(obs_diff > 0),
                effect=f"Δr={obs_diff:+.3f} "
                       f"(fixed r={corrs_fixed['lambda_j']['r']:.3f} → "
                       f"varied r={corrs_varied['lambda_j']['r']:.3f})")

    # ---- Exploratory: per-axis RMSE and bias ----
    for cond in ("fixed", "varied"):
        for d, ax in enumerate(THETA_AXES):
            err = out[cond]["hat"][:, d] - out[cond]["true"][:, d]
            t = one_sample_perm(err, 0.0)
            reg.explore("H6", f"recovery ({cond}) {AXIS_LABEL_KO[ax]} "
                        f"bias != 0", t["p"],
                        effect=f"bias={err.mean():+.3f}, "
                               f"RMSE={np.sqrt(np.mean(err ** 2)):.3f}")

    out["corr_fixed"] = corrs_fixed
    out["corr_varied"] = corrs_varied
    return out


# ================================================================ Task B
def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """
    AUC of the predicted probability p against the binary label
    y (1 = cooperate), computed via the Mann-Whitney U statistic
    (ties handled by mean ranks).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # Tie correction
    uniq, inv_idx, cnt = np.unique(p, return_inverse=True, return_counts=True)
    for u_i in np.where(cnt > 1)[0]:
        m = inv_idx == u_i
        ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


#: Axes confirmed for projection fidelity. omega and eta are only
#: weakly identified from one's own action history (the focal does not
#: generate g-axis variation by its own policy), so they are
#: exploratory.
FIDELITY_AXES = ("alpha", "rho", "beta", "lambda_j")


def run_projection(cfg: Config, reg: Registry) -> dict:
    """
    Task B — validity of self-projection.

    B1 (confirmatory) : in a HalloReg x HalloReg dyad, the fidelity
                        between A's theta-hat_self and B's
                        theta-hat(A).
    B2 (exploratory)  : calibration of the self-projection filter
                        against one's own behaviour (AUC, Brier).
    """
    hk = cfg.halloreg_kwargs()

    # ---------- B1: projection fidelity (HalloReg vs HalloReg) ----
    specs_f = []
    for sd in range(cfg.seeds):
        specs_f.append({
            "agent": {"type": "halloreg", "seed": 40_000 + sd * 43, **hk},
            "opponent": {"type": "halloreg", "seed": 41_000 + sd * 43, **hk},
            "env_err_agent": cfg.env_error, "env_err_opponent": cfg.env_error,
            "noise_seed": 820_000 + sd * 109})
    res_f = run_many(specs_f, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                     desc="H6 projection-fidelity dyads")

    axes = list(THETA_AXES)
    proj = np.zeros((cfg.seeds, len(axes)))     # A's theta-hat_self
    partner = np.zeros((cfg.seeds, len(axes)))  # B's inference of A
    for sd in range(cfg.seeds):
        la = res_f[sd]["agent_log"]
        lb = res_f[sd]["opponent_log"]
        for d, ax in enumerate(axes):
            proj[sd, d] = float(la[f"P_{ax}"][-1])
            partner[sd, d] = float(lb[f"E_{ax}"][-1])

    fidelity = {}
    for d, ax in enumerate(axes):
        c = corr_boot(proj[:, d], partner[:, d], seed=200 + d)
        fidelity[ax] = c
        label = (f"projection fidelity {AXIS_LABEL_KO[ax]}: "
                 f"corr(theta_self, theta_partner) > 0")
        if ax in FIDELITY_AXES:
            reg.confirm("H6", label, c["p"], direction_ok=bool(c["r"] > 0),
                        effect=f"r={c['r']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]")
        else:
            reg.explore("H6", label + " (weakly identified from own "
                        "behaviour alone)", c["p"],
                        effect=f"r={c['r']:.3f}")
    # Per-axis absolute deviation (interpretive aid)
    for d, ax in enumerate(axes):
        err = proj[:, d] - partner[:, d]
        t = one_sample_perm(err, 0.0)
        reg.explore("H6", f"projection bias {AXIS_LABEL_KO[ax]} != 0",
                    t["p"],
                    effect=f"bias={err.mean():+.3f}, "
                           f"RMSE={np.sqrt(np.mean(err ** 2)):.3f}")

    # ---------- B2: predictive calibration (exploratory) ----------
    specs, registry = [], {}
    for ti, tname in enumerate(ALL_TYPES):
        for sd in range(cfg.seeds):
            opp = dict(type_spec(tname))
            opp["seed"] = 31_000 + sd * 41 + ti
            if tname == "halloreg":
                opp.update(hk)
            registry[(ti, sd)] = len(specs)
            specs.append({
                "agent": {"type": "halloreg", "seed": 30_000 + sd * 41 + ti, **hk},
                "opponent": opp,
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 810_000 + sd * 107 + ti})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H6 projection-calibration dyads")

    auc_by_type: Dict[str, np.ndarray] = {}
    skill_by_type: Dict[str, np.ndarray] = {}
    for ti, tname in enumerate(ALL_TYPES):
        aucs = np.full(cfg.seeds, np.nan)
        skills = np.full(cfg.seeds, np.nan)
        for sd in range(cfg.seeds):
            log = res[registry[(ti, sd)]]["agent_log"]
            p = np.asarray(log["self_pred_coop"], dtype=float)[1:]
            y = (np.asarray(log["action"], dtype=int) == COOP).astype(int)[1:]
            ok = np.isfinite(p)
            p, y = p[ok], y[ok]
            if len(y) < 5 or y.sum() in (0, len(y)):
                continue      # AUC undefined if behaviour is degenerate
            aucs[sd] = _auc(y, p)
            # Causal base-rate baseline: cumulative cooperation rate
            # up to that round
            base = np.concatenate([[0.5], np.cumsum(y)[:-1]
                                   / np.arange(1, len(y))])
            bm = float(np.mean((p - y) ** 2))
            bb = float(np.mean((base - y) ** 2))
            skills[sd] = 1.0 - bm / max(bb, 1e-9)
        auc_by_type[tname] = aucs
        skill_by_type[tname] = skills

    all_auc = np.concatenate([auc_by_type[t] for t in ALL_TYPES])
    all_auc = all_auc[np.isfinite(all_auc)]
    all_skill = np.concatenate([skill_by_type[t] for t in ALL_TYPES])
    all_skill = all_skill[np.isfinite(all_skill)]
    t1 = one_sample_perm(all_auc, 0.5, alternative="greater")
    c1 = boot_mean_ci(all_auc)
    reg.explore("H6", "self-projection predictive AUC > 0.5 (much "
                      "round-level variation is policy sampling noise "
                      "— near 0.5 predicted)",
                t1["p"],
                effect=f"AUC={c1['mean']:.3f} "
                       f"[{c1['ci'][0]:.3f}, {c1['ci'][1]:.3f}]")
    t2 = one_sample_perm(all_skill, 0.0, alternative="greater")
    c2 = boot_mean_ci(all_skill)
    reg.explore("H6", "self-projection Brier skill score > 0 (vs the "
                "base-rate forecast)", t2["p"],
                effect=f"skill={c2['mean']:+.3f} "
                       f"[{c2['ci'][0]:+.3f}, {c2['ci'][1]:+.3f}]")

    return {"axes": axes, "proj": proj, "partner": partner,
            "fidelity": fidelity,
            "auc": auc_by_type, "skill": skill_by_type,
            "mean_auc": float(c1["mean"]), "mean_skill": float(c2["mean"])}


# =================================================================== Run
def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H6] parameter recovery and self-projection")
    a = run_recovery(cfg, reg)
    b = run_projection(cfg, reg)
    out = {"recovery": a, "projection": b}
    _plot(cfg, a, b)
    save_json({"recovery": {
        "corr_fixed": {k: {kk: v[kk] for kk in ("r", "ci", "p", "n")}
                       for k, v in a["corr_fixed"].items()},
        "corr_varied": {k: {kk: v[kk] for kk in ("r", "ci", "p", "n")}
                        for k, v in a["corr_varied"].items()}},
        "projection": {
            "fidelity": {k: {kk: v[kk] for kk in ("r", "ci", "p", "n")}
                         for k, v in b["fidelity"].items()},
            "mean_auc": b["mean_auc"], "mean_skill": b["mean_skill"],
            "auc_by_type": {t: float(np.nanmean(b["auc"][t]))
                            for t in ALL_TYPES}}},
        cfg.results / "H6.json")
    return out


# ============================================================== Figures
def _plot(cfg: Config, a: dict, b: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import annotate_n, band_plot

    axes = list(THETA_AXES)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

    # (a-c) scatter of the main axes in the fixed condition
    for i, ax_name in enumerate(("alpha", "rho", "eta")):
        ax = fig.add_subplot(gs[0, i])
        d = axes.index(ax_name)
        x = a["fixed"]["true"][:, d]; y = a["fixed"]["hat"][:, d]
        ax.scatter(x, y, s=16, alpha=0.6, color="#4C72B0")
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, ls="--", color="#888888", lw=1)
        ax.set_xlabel(f"true {AXIS_LABEL_KO[ax_name]}")
        ax.set_ylabel(f"estimated {AXIS_LABEL_KO[ax_name]}")
        r = a["corr_fixed"][ax_name]["r"]
        ax.set_title(f"({'abc'[i]}) recovery (fixed) — "
                     f"{AXIS_LABEL_KO[ax_name]}\nr={r:.3f}")
        annotate_n(ax, len(x))

    # (d) per-axis recovery correlation: fixed vs varied
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(axes)); w = 0.36
    rf = [a["corr_fixed"][k]["r"] for k in axes]
    rv = [a["corr_varied"][k]["r"] for k in axes]
    ax.bar(x - w / 2, rf, w, color="#4C72B0", label="fixed (standard PD)",
           alpha=0.9)
    ax.bar(x + w / 2, rv, w, color="#C44E52",
           label="varied (payoff modulation)",
           alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABEL_KO[k] for k in axes], rotation=40,
                       ha="right", fontsize=7)
    ax.set_ylabel("recovery correlation r"); ax.set_ylim(-0.2, 1.05)
    ax.set_title("(d) recovery correlation by axis\n"
                 "lambda_j is collinear with alpha under fixed payoffs; "
                 "modulation identifies it")
    ax.legend(fontsize=7)

    # (e) lambda_j recovery: scatter for both conditions
    ax = fig.add_subplot(gs[1, 1])
    d = axes.index("lambda_j")
    ax.scatter(a["fixed"]["true"][:, d], a["fixed"]["hat"][:, d], s=15,
               alpha=0.55, color="#4C72B0", label="fixed")
    ax.scatter(a["varied"]["true"][:, d], a["varied"]["hat"][:, d], s=15,
               alpha=0.55, color="#C44E52", label="varied")
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1)
    ax.set_xlabel("true lambda_j"); ax.set_ylabel("estimated lambda_j")
    ax.set_title("(e) lambda_j recovery — identifiability contrast")
    ax.legend(fontsize=7)

    # (f) recovery convergence (absolute error against the truth)
    ax = fig.add_subplot(gs[1, 2])
    for ax_name, col in (("alpha", "#4C72B0"), ("rho", "#55A868"),
                         ("beta", "#C44E52"), ("eta", "#8172B2")):
        d = axes.index(ax_name)
        err = np.abs(a["fixed"]["traj"][:, :, d]
                     - a["fixed"]["true"][:, None, d])
        # Axes differ in scale, so normalise by the true range
        rng_ = TRUE_RANGE[ax_name][1] - TRUE_RANGE[ax_name][0]
        ax.plot(np.median(err, axis=0) / rng_, color=col,
                label=AXIS_LABEL_KO[ax_name], lw=1.3)
    ax.set_xlabel("round")
    ax.set_ylabel("normalised absolute error (median)")
    ax.set_title("(f) recovery convergence")
    ax.legend(fontsize=7)

    # (g) projection fidelity — alpha axis scatter
    ax = fig.add_subplot(gs[2, 0])
    d = b["axes"].index("alpha")
    ax.scatter(b["partner"][:, d], b["proj"][:, d], s=16, alpha=0.6,
               color="#8172B2")
    lim = [min(b["partner"][:, d].min(), b["proj"][:, d].min()),
           max(b["partner"][:, d].max(), b["proj"][:, d].max())]
    ax.plot(lim, lim, ls="--", color="#888888", lw=1)
    ax.set_xlabel("alpha inferred about me by the partner")
    ax.set_ylabel("my self-projected alpha")
    ax.set_title(f"(g) projection fidelity — alpha axis\n"
                 f"r={b['fidelity']['alpha']['r']:.3f}")
    annotate_n(ax, b["proj"].shape[0])

    # (h) projection fidelity correlation by axis
    ax = fig.add_subplot(gs[2, 1])
    ax_names = b["axes"]
    x = np.arange(len(ax_names))
    rs = [b["fidelity"][k]["r"] for k in ax_names]
    cols = ["#8172B2" if k in FIDELITY_AXES else "#BBBBBB" for k in ax_names]
    ax.bar(x, rs, color=cols, alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABEL_KO[k] for k in ax_names], rotation=40,
                       ha="right", fontsize=7)
    ax.set_ylabel("corr(θ̂_self, θ̂_partner)"); ax.set_ylim(-0.4, 1.05)
    ax.set_title("(h) projection fidelity by axis\n"
                 "(grey = weakly identified from own behaviour)")

    # (i) per-axis RMSE summary
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(len(axes)); w = 0.36
    for off, cond, col in ((-w / 2, "fixed", "#4C72B0"),
                           (+w / 2, "varied", "#C44E52")):
        rm = []
        for d, k in enumerate(axes):
            e = a[cond]["hat"][:, d] - a[cond]["true"][:, d]
            rng_ = TRUE_RANGE[k][1] - TRUE_RANGE[k][0]
            rm.append(float(np.sqrt(np.mean(e ** 2)) / rng_))
        ax.bar(x + off, rm, w, color=col, label=cond, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABEL_KO[k] for k in axes], rotation=40,
                       ha="right", fontsize=7)
    ax.set_ylabel("normalised RMSE")
    ax.set_title("(i) recovery error by axis"); ax.legend(fontsize=7)

    fig.suptitle("H6 — robustness of parameter recovery and "
                 "self-projection", fontsize=12, y=0.985)
    save_fig(fig, cfg, "H6_recovery_projection")
