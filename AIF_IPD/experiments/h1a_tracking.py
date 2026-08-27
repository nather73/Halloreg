"""
experiments.h1a_tracking
========================

**H1A — does OpponentInversion track another agent's *changing*
intent, and is lambda itself recovered?**

Where H1 classifies static types, H1A tracks dynamic traits, split
into two independent subtasks.

Task A — intent tracking
------------------------
The opponent is a `StrategyAgent` whose disposition switches every
SWITCH_PERIOD rounds (e.g. TFT -> ALLD -> GTFT -> TFT). Execution
noise is applied as well, so "noise" and "intent change" coexist.

Index — the **switch-aligned tracking index**, defined not as a
projection of theta-hat onto a cooperativeness axis but as the
**correlation between the true cooperation tendency of the active
phase and the cooperation probability predicted by the inference**.
Each round's `pred_coop` is matched against the true cooperation rate
of the phase active in that round — the most direct
operationalisation of "tracking": in a cooperative phase the
inference should predict a high cooperation probability.

  - Confirmatory A : per-seed correlation r > 0 (sign-flip
                     permutation test).
  - Exploratory A  : post-switch recovery latency — rounds until
                     |delta pred_coop| exceeds half the pre/post
                     difference.

Task B — lambda recovery
------------------------
With an **EmpathicAgent (fixed lambda)** as the opponent the true
lambda is known, and switching it mid-session (lam_schedule)
creates "changing empathy".

  Condition 1 (static)  : lambda_true in {0.0 ... 1.0}, each fixed.
                          Correlation of the final lambda_j-hat with
                          lambda_true = recovery accuracy.
  Condition 2 (dynamic) : lambda switches low<->high every
                          SWITCH_PERIOD rounds. Round-wise
                          correlation = tracking accuracy.

  - Confirmatory B1 : static — corr(lambda_j-hat, lambda_true) > 0.
  - Confirmatory B2 : dynamic — within-seed
                      corr(lambda_j-hat(t), lambda_true(t)) > 0.

[Interpretive caveat — stated explicitly]
lambda_j enters the opponent's behaviour only as an intercept,
s(lambda_j, p) = (T-S)*lambda_j + ..., so alpha and lambda_j are
partially collinear under fixed payoffs (both raise the intercept of
the cooperation probability). This collinearity is a limitation of
the present design and **predicts in advance** that recovery
correlations will be imperfect. H6 runs a separate design that breaks
it by modulating (T, S) per block; here the recoverability under the
experimental condition (fixed PD) is reported as it is.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import SWITCH_PERIOD, SWITCH_SCENARIOS, switch_rounds
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, corr_boot, one_sample_perm,
)
from AIF_IPD.ipd.sim import run_many
from .common import Config, Registry, save_fig, save_json

LOGGER = get_logger("HalloReg.H1A")

#: Grid of true lambda values used in the static recovery condition
LAM_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: Low/high levels of the dynamic lambda switch
LAM_LO, LAM_HI = 0.1, 0.9

#: The 'true cooperation tendency' of each fixed-strategy phase (the
#: reference signal for switch tracking): the representative
#: probability that the opponent cooperates in response to a
#: cooperative focal. ALLD is 0, ALLC is 1, reciprocal strategies are
#: high while the focal cooperates.
PHASE_COOP = {"tft": 0.9, "gtft": 0.95, "wsls": 0.7,
              "allc": 1.0, "alld": 0.0, "random": 0.5}


def _safe_mean(x) -> float:
    """
    Mean excluding NaN; NaN if no finite value remains.

    Per-seed correlation arrays are NaN where the signal was constant
    in that seed (e.g. a horizon too short for any phase switch makes
    the reference constant). np.nanmean warns in that case, so this is
    handled explicitly to keep the logs clean.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


# ================================================================ Task A
def _phase_signal(scenario: str, n_rounds: int) -> np.ndarray:
    """True cooperation tendency of the active phase per round."""
    cycle = SWITCH_SCENARIOS[scenario]
    return np.array([PHASE_COOP[cycle[(t // SWITCH_PERIOD) % len(cycle)]]
                     for t in range(n_rounds)])


def run_tracking(cfg: Config, reg: Registry) -> dict:
    """
    Task A — tracking trait switches.

    [Scale requirement] The switch period is SWITCH_PERIOD, so if
    cfg.rounds is at or below it no phase switch occurs within a
    session; the reference signal is then constant, the correlation is
    undefined (r = 0) and the test is meaningless. A "not supported"
    verdict at smoke scale is due to that, not to a model failure —
    warned in the log.
    """
    if cfg.rounds < 2 * SWITCH_PERIOD:
        LOGGER.warning(
            "[H1A] rounds=%d < 2*SWITCH_PERIOD(%d) — no phase switch, "
            "so the tracking correlation is undefined. This test is "
            "valid only for rounds >= %d.",
            cfg.rounds, SWITCH_PERIOD, 2 * SWITCH_PERIOD)
    scenarios = list(SWITCH_SCENARIOS)
    hk = cfg.halloreg_kwargs()

    specs, registry = [], {}
    for si, sc in enumerate(scenarios):
        for sd in range(cfg.seeds):
            registry[(si, sd)] = len(specs)
            specs.append({
                "agent": {"type": "halloreg", "seed": 3000 + sd * 19 + si, **hk},
                # The switching opponent is a StrategyAgent with a
                # schedule.
                "opponent": {"type": "strategy",
                             "kind": SWITCH_SCENARIOS[sc][0],
                             "seed": 4000 + sd * 19 + si,
                             "error": 0.05,
                             "schedule": [
                                 (r, SWITCH_SCENARIOS[sc][
                                     (r // SWITCH_PERIOD) % len(SWITCH_SCENARIOS[sc])])
                                 for r in range(0, cfg.rounds, SWITCH_PERIOD)]},
                "env_err_agent": cfg.env_error, "env_err_opponent": 0.0,
                "noise_seed": 610_000 + sd * 71 + si,
            })

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H1A trait-switch dyads")

    corr_by_scenario: Dict[str, np.ndarray] = {}
    traces: Dict[str, np.ndarray] = {}
    for si, sc in enumerate(scenarios):
        truth = _phase_signal(sc, cfg.rounds)
        rs = np.full(cfg.seeds, np.nan)
        tr = np.zeros((cfg.seeds, cfg.rounds))
        for sd in range(cfg.seeds):
            pc = np.asarray(res[registry[(si, sd)]]["agent_log"]["pred_coop"],
                            dtype=float)
            tr[sd] = pc
            if pc.std() > 0 and truth.std() > 0:
                rs[sd] = float(np.corrcoef(pc, truth)[0, 1])
        corr_by_scenario[sc] = rs
        traces[sc] = tr

    all_r = np.concatenate([corr_by_scenario[s] for s in scenarios])
    t = one_sample_perm(all_r, 0.0, alternative="greater")
    ci = boot_mean_ci(all_r)
    reg.confirm("H1A", "trait-switch tracking: "
                "corr(predicted cooperation, true phase) > 0",
                t["p"], direction_ok=bool(_safe_mean(all_r) > 0),
                effect=f"r̄={ci['mean']:.3f} [{ci['ci'][0]:.3f}, {ci['ci'][1]:.3f}]",
                detail={"mean_r": ci["mean"], "ci": ci["ci"]})

    # Exploratory: tracking correlation per scenario
    for sc in scenarios:
        rs = corr_by_scenario[sc]
        tt = one_sample_perm(rs, 0.0, alternative="greater")
        c = boot_mean_ci(rs)
        reg.explore("H1A", f"scenario {sc} tracking correlation",
                    tt["p"],
                    effect=f"r̄={c['mean']:.3f}")

    return {"scenarios": scenarios,
            "corr": {s: corr_by_scenario[s] for s in scenarios},
            "mean_corr": float(_safe_mean(all_r)),
            "traces": traces,
            "truth": {s: _phase_signal(s, cfg.rounds) for s in scenarios},
            "switch_rounds": switch_rounds(cfg.rounds)}


# ================================================================ Task B
def run_lambda_recovery(cfg: Config, reg: Registry) -> dict:
    """Task B — lambda recovery (static and dynamic)."""
    hk = cfg.halloreg_kwargs()
    ek = {"n_particles": cfg.n_particles, "planning_horizon": cfg.horizon}

    # ---- Condition 1: static lambda ----
    specs, reg_static = [], {}
    for li, lam in enumerate(LAM_GRID):
        for sd in range(cfg.seeds):
            reg_static[(li, sd)] = len(specs)
            specs.append({
                "agent": {"type": "halloreg", "seed": 5000 + sd * 23 + li, **hk},
                "opponent": {"type": "empathic", "lam": float(lam),
                             "seed": 6000 + sd * 23 + li, **ek},
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 620_000 + sd * 83 + li})
    n_static = len(specs)

    # ---- Condition 2: dynamic lambda (low<->high each period) ----
    lam_sched = [(r, LAM_LO if (r // SWITCH_PERIOD) % 2 == 0 else LAM_HI)
                 for r in range(0, cfg.rounds, SWITCH_PERIOD)]
    reg_dyn = {}
    for sd in range(cfg.seeds):
        reg_dyn[sd] = len(specs)
        specs.append({
            "agent": {"type": "halloreg", "seed": 7000 + sd * 29, **hk},
            "opponent": {"type": "empathic", "lam": LAM_LO,
                         "lam_schedule": lam_sched,
                         "seed": 8000 + sd * 29, **ek},
            "env_err_agent": cfg.env_error, "env_err_opponent": cfg.env_error,
            "noise_seed": 630_000 + sd * 89})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H1A lambda-recovery dyads")

    # ---- Static condition analysis ----
    lam_hat = np.zeros((len(LAM_GRID), cfg.seeds))
    for li in range(len(LAM_GRID)):
        for sd in range(cfg.seeds):
            lam_hat[li, sd] = float(
                res[reg_static[(li, sd)]]["agent_log"]["E_lambda_j"][-1])

    x = np.repeat(np.array(LAM_GRID), cfg.seeds)
    y = lam_hat.reshape(-1)
    cs = corr_boot(x, y)
    reg.confirm("H1A", "static lambda recovery: "
                "corr(lambda_j-hat, lambda_true) > 0", cs["p"],
                direction_ok=bool(cs["r"] > 0),
                effect=f"r={cs['r']:.3f} [{cs['ci'][0]:.3f}, {cs['ci'][1]:.3f}]",
                detail=cs)

    # ---- Dynamic condition: within-seed round-wise correlation ----
    lam_true_t = np.array([LAM_LO if (t // SWITCH_PERIOD) % 2 == 0 else LAM_HI
                           for t in range(cfg.rounds)])
    dyn_r = np.full(cfg.seeds, np.nan)
    dyn_traces = np.zeros((cfg.seeds, cfg.rounds))
    for sd in range(cfg.seeds):
        lj = np.asarray(res[reg_dyn[sd]]["agent_log"]["E_lambda_j"], dtype=float)
        dyn_traces[sd] = lj
        if lj.std() > 0:
            dyn_r[sd] = float(np.corrcoef(lj, lam_true_t)[0, 1])
    td = one_sample_perm(dyn_r, 0.0, alternative="greater")
    cd = boot_mean_ci(dyn_r)
    reg.confirm("H1A", "dynamic lambda tracking: within-seed "
                "corr(lambda_j-hat(t), lambda_true(t)) > 0",
                td["p"], direction_ok=bool(cd["mean"] > 0),
                effect=f"r̄={cd['mean']:.3f} [{cd['ci'][0]:.3f}, {cd['ci'][1]:.3f}]",
                detail={"mean_r": cd["mean"], "ci": cd["ci"]})

    return {"lam_grid": list(LAM_GRID), "lam_hat": lam_hat,
            "static_corr": cs, "dyn_corr": dyn_r,
            "dyn_traces": dyn_traces, "lam_true_t": lam_true_t,
            "n_static_specs": n_static}


# =================================================================== Run
def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H1A] tracking changing intent and recovering lambda")
    a = run_tracking(cfg, reg)
    b = run_lambda_recovery(cfg, reg)
    out = {"tracking": a, "lambda_recovery": b}
    _plot(cfg, a, b)
    save_json(out, cfg.results / "H1A.json")
    return out


# ============================================================== Figures
def _plot(cfg: Config, a: dict, b: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import band_plot, annotate_n

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.30)
    scenarios = a["scenarios"]

    # (a-c) predicted cooperation trajectories vs the true phase
    for i, sc in enumerate(scenarios[:3]):
        ax = fig.add_subplot(gs[0, i])
        band_plot(ax, a["traces"][sc], color="#4C72B0",
                  label="inferred cooperation probability")
        ax.plot(a["truth"][sc], color="crimson", lw=1.4, ls="--",
                label="true phase tendency")
        for r in a["switch_rounds"]:
            ax.axvline(r, color="#999999", lw=0.7, ls=":")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("round"); ax.set_ylabel("cooperation probability")
        ax.set_title(f"({'abc'[i]}) trait-switch tracking — {sc}\n"
                     f"r̄={_safe_mean(a['corr'][sc]):.3f}")
        if i == 0:
            ax.legend(loc="lower right", fontsize=7)
        annotate_n(ax, cfg.seeds)

    # (d) static lambda recovery scatter
    ax = fig.add_subplot(gs[1, 0])
    grid = b["lam_grid"]
    m = b["lam_hat"].mean(axis=1)
    s = b["lam_hat"].std(axis=1, ddof=1)
    for li, lam in enumerate(grid):
        ax.scatter(np.full(b["lam_hat"].shape[1], lam), b["lam_hat"][li],
                   s=5, alpha=0.18, color="#4C72B0")
    ax.errorbar(grid, m, yerr=s, fmt="o-", color="crimson", capsize=3, lw=1.5,
                label="seed mean +/- SD")
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1,
            label="perfect recovery")
    ax.set_xlabel("true lambda (opponent's fixed empathy)")
    ax.set_ylabel("estimated lambda_j-hat (final)")
    ax.set_title(f"(d) static lambda recovery — "
                 f"r={b['static_corr']['r']:.3f}")
    ax.legend(fontsize=7)

    # (e) dynamic lambda tracking trajectories
    ax = fig.add_subplot(gs[1, 1])
    band_plot(ax, b["dyn_traces"], color="#55A868",
              label="estimated lambda_j-hat")
    ax2 = ax.twinx()
    ax2.plot(b["lam_true_t"], color="crimson", ls="--", lw=1.4,
             label="true lambda(t)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("true lambda", color="crimson")
    ax2.grid(False)
    ax.set_xlabel("round"); ax.set_ylabel("estimated lambda_j-hat")
    ax.set_title(f"(e) dynamic lambda tracking — "
                 f"mean r={_safe_mean(b['dyn_corr']):.3f}")
    ax.legend(loc="upper left", fontsize=7)
    annotate_n(ax, cfg.seeds)

    # (f) per-seed distribution of tracking correlations
    ax = fig.add_subplot(gs[1, 2])
    data = [a["corr"][s] for s in scenarios] + [b["dyn_corr"]]
    labels = list(scenarios) + ["lambda tracking"]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8172B2"); patch.set_alpha(0.7)
    ax.axhline(0, color="crimson", ls="--", lw=1.2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("per-seed correlation")
    ax.set_title("(f) seed distribution of tracking correlations\n"
                 "(0 = no tracking)")

    fig.suptitle("H1A — tracking changing intent and recovering "
                 "lambda", fontsize=12, y=0.98)
    save_fig(fig, cfg, "H1A_tracking_recovery")
