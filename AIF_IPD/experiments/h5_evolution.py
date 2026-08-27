"""
experiments.h5_evolution
========================

**H5 — in a mixed population under non-stationary payoffs, does
HalloReg survive across generations under replicator (RE) and optimal
replicator (ORE) dynamics?**

Design
------
The per-regime payoff matrices Pi already estimated in H4 are reused
(no re-simulation). Initial compositions x0 come from the **same
composition space** as H3/H4: x0 = counts / 30.

For each (regime x seed x composition):
  1. Integrate RE to obtain the terminal composition x_RE(tau).
  2. Integrate ORE to obtain x_ORE(tau).
  3. Survival = [x_HalloReg(tau) > 1/30] (the share of one
     individual).

Indices and tests
-----------------
- Survival fraction — the share of initial compositions from which
  the type survives.
- Dominance fraction — the share where it holds the largest terminal
  frequency.
- Mean terminal frequency.

  - Confirmatory H5-1 : HalloReg survival under RE > chance baseline.
  - Confirmatory H5-2 : HalloReg survival under ORE > chance
                        baseline.

**Definition of the chance baseline.** Survival differs in difficulty
across types, so an absolute threshold is hard to compare against.
The baseline here is the mean survival rate across the six types (the
observed mean, not 1/6), which asks "does HalloReg survive from a
wider set of initial conditions than the average type?". The baseline
is data-dependent, but the test is valid because it is paired at the
seed level.

  - Exploratory : per-regime survival, survival vs ALLD, the RE->ORE
                  change, and terminal CC rate.

Managing the computation
------------------------
With over a hundred thousand compositions, running ORE on all of them
is impractical. ORE is fully vectorised along the initial-point axis
and can be batched, but the FBSM repeats sweeps x steps, making it
tens of times more expensive than RE. Therefore:

  - RE  : `n_init` compositions are sampled at random from the
          composition space, **shared across regimes and seeds**
          (common random numbers, so comparisons stay paired).
  - ORE : the same set of initial points (keeping RE and ORE paired).

n_init defaults to 1500, a stratified sample of the composition space
sufficient for basin estimates.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import ALL_TYPES, TYPE_LABEL_KO
from AIF_IPD.ipd.evolution import (
    dominance_fraction, mean_terminal_frequency, ore_ends_batch,
    replicator_ends_batch, survival_fraction, terminal_cc,
)
from AIF_IPD.ipd.metrics import boot_mean_ci, effect_size_paired, fmt_es, one_sample_perm
from AIF_IPD.ipd.payoff_schedule import REGIME_LABEL_KO
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json

LOGGER = get_logger("HalloReg.H5")

HR = ALL_TYPES.index("halloreg")
SURVIVAL_THRESHOLD = 1.0 / 30.0     # one individual in a population of 30

#: Number of initial points used for basin estimates (sampled from
#: the composition space)
N_INIT_FULL = 1500
N_INIT_QUICK = 200

#: Evolutionary integration parameters
RE_STEPS = 600
ORE_TAU, ORE_STEPS, ORE_SWEEPS = 1.0, 200, 24


def _sample_initials(comps: np.ndarray, n: int, seed: int = 5) -> np.ndarray:
    """Uniformly sample initial compositions x0 = counts/30 from the
    composition space."""
    rng = np.random.default_rng(seed)
    if len(comps) <= n:
        sel = comps
    else:
        sel = comps[rng.choice(len(comps), size=n, replace=False)]
    X0 = sel.astype(float)
    return X0 / X0.sum(axis=1, keepdims=True)


def run(cfg: Config, reg: Registry, h34: dict, comps: np.ndarray) -> dict:
    """
    Run H5. `h34` is the return value of h3_h4_population.run(); its
    payoff matrices are reused.
    """
    n_init = N_INIT_QUICK if cfg.quick else N_INIT_FULL
    X0 = _sample_initials(comps, n_init)
    LOGGER.info("[H5] evolutionary dynamics — %d initial points x "
                "%d regimes x %d seeds",
                len(X0), len(h34["nonstationary"]["regimes"]), cfg.seeds)

    regimes = h34["nonstationary"]["regimes"]
    k = len(ALL_TYPES)

    # Many seeds would make the evolutionary integration excessive, so
    # only as many as the tests need are used. (Pi is already a mean
    # over `seeds` dyads, so seed-level variation is small.)
    n_seed_eval = min(cfg.seeds, 24 if not cfg.quick else 4)

    results: Dict[str, dict] = {}
    for rg in regimes:
        pair = h34["nonstationary"]["per_regime"][rg]["pair"]
        surv_re = np.zeros((n_seed_eval, k))
        surv_ore = np.zeros((n_seed_eval, k))
        dom_re = np.zeros((n_seed_eval, k))
        dom_ore = np.zeros((n_seed_eval, k))
        freq_re = np.zeros((n_seed_eval, k))
        freq_ore = np.zeros((n_seed_eval, k))
        cc_re = np.zeros(n_seed_eval)
        cc_ore = np.zeros(n_seed_eval)

        for s in range(n_seed_eval):
            Pi = pair["Pi_raw"][:, :, s]
            CCm = pair["CCm_raw"][:, :, s]
            ends_re = replicator_ends_batch(Pi, X0, steps=RE_STEPS)
            ends_ore = ore_ends_batch(Pi, X0, tau=ORE_TAU, steps=ORE_STEPS,
                                      sweeps=ORE_SWEEPS)
            for i in range(k):
                surv_re[s, i] = survival_fraction(ends_re, i, SURVIVAL_THRESHOLD)
                surv_ore[s, i] = survival_fraction(ends_ore, i, SURVIVAL_THRESHOLD)
                dom_re[s, i] = dominance_fraction(ends_re, i)
                dom_ore[s, i] = dominance_fraction(ends_ore, i)
                freq_re[s, i] = mean_terminal_frequency(ends_re, i)
                freq_ore[s, i] = mean_terminal_frequency(ends_ore, i)
            cc_re[s] = terminal_cc(CCm, ends_re)
            cc_ore[s] = terminal_cc(CCm, ends_ore)

        # Representative trajectories (for the figures), starting from
        # a uniform initial composition
        from AIF_IPD.ipd.evolution import ore_trajectory, replicator_trajectory
        x_unif = np.full(k, 1.0 / k)
        traj_re = replicator_trajectory(pair["Pi"], x_unif, steps=RE_STEPS)
        traj_ore = ore_trajectory(pair["Pi"], x_unif, tau=ORE_TAU,
                                  steps=400, sweeps=40)["x"]

        results[rg] = {
            "surv_re": surv_re, "surv_ore": surv_ore,
            "dom_re": dom_re, "dom_ore": dom_ore,
            "freq_re": freq_re, "freq_ore": freq_ore,
            "cc_re": cc_re, "cc_ore": cc_ore,
            "traj_re": traj_re, "traj_ore": traj_ore,
        }
        LOGGER.info("  [%s] survival RE: HalloReg=%.3f "
                    "(all-type mean=%.3f) | ORE: HalloReg=%.3f "
                    "(mean=%.3f)",
                    rg, surv_re[:, HR].mean(), surv_re.mean(),
                    surv_ore[:, HR].mean(), surv_ore.mean())

    # ---- Confirmatory: pooled across regimes, HalloReg survival >
    # the all-type mean ----
    for tag, key in (("RE", "surv_re"), ("ORE", "surv_ore")):
        hr = np.concatenate([results[rg][key][:, HR] for rg in regimes])
        avg = np.concatenate([results[rg][key].mean(axis=1) for rg in regimes])
        d = hr - avg
        t = one_sample_perm(d, 0.0, alternative="greater")
        e = effect_size_paired(d)
        ci = boot_mean_ci(d)
        reg.confirm("H5", f"{tag} survival basin: HalloReg > all-type "
                    f"mean", t["p"],
                    direction_ok=bool(ci["mean"] > 0),
                    effect=f"Δ={ci['mean']:+.4f} "
                           f"[{ci['ci'][0]:+.4f}, {ci['ci'][1]:+.4f}], "
                           f"{fmt_es(e, 'dz')}; "
                           f"HalloReg={hr.mean():.3f}",
                    detail={"halloreg": float(hr.mean()),
                            "all_type_mean": float(avg.mean())})

    # ---- Exploratory: per regime, per rival, ORE-RE difference,
    # terminal CC ----
    for rg in regimes:
        for tag, key in (("RE", "surv_re"), ("ORE", "surv_ore")):
            hr = results[rg][key][:, HR]
            avg = results[rg][key].mean(axis=1)
            t = one_sample_perm(hr - avg, 0.0, alternative="greater")
            reg.explore("H5", f"[{REGIME_LABEL_KO[rg]}] {tag} survival "
                        f"rate", t["p"],
                        effect=f"HalloReg={hr.mean():.3f} vs "
                               f"mean={avg.mean():.3f}")
    for i, t_ in enumerate(ALL_TYPES):
        if i == HR:
            continue
        d = np.concatenate([results[rg]["surv_ore"][:, HR]
                            - results[rg]["surv_ore"][:, i] for rg in regimes])
        tt = one_sample_perm(d, 0.0, alternative="greater")
        reg.explore("H5", f"ORE survival rate: HalloReg vs "
                    f"{TYPE_LABEL_KO[t_]}",
                    tt["p"], effect=f"Δ={np.mean(d):+.4f}")
    d_cc = np.concatenate([results[rg]["cc_ore"] - results[rg]["cc_re"]
                           for rg in regimes])
    t = one_sample_perm(d_cc, 0.0, alternative="greater")
    reg.explore("H5", "terminal CC rate: ORE > RE (group-level "
                "selection promotes cooperation)", t["p"],
                effect=f"Δ={np.mean(d_cc):+.4f}")

    out = {"regimes": regimes, "results": results, "n_init": int(len(X0)),
           "n_seed_eval": n_seed_eval, "threshold": SURVIVAL_THRESHOLD}
    _plot(cfg, out)
    save_json({"regimes": regimes, "n_init": int(len(X0)),
               "summary": {rg: {
                   "surv_re": results[rg]["surv_re"].mean(axis=0),
                   "surv_ore": results[rg]["surv_ore"].mean(axis=0),
                   "dom_ore": results[rg]["dom_ore"].mean(axis=0),
                   "freq_ore": results[rg]["freq_ore"].mean(axis=0),
                   "cc_re": float(results[rg]["cc_re"].mean()),
                   "cc_ore": float(results[rg]["cc_ore"].mean())}
                   for rg in regimes},
               "types": list(ALL_TYPES)},
              cfg.results / "H5.json")
    return out


# ============================================================== Figures
def _plot(cfg: Config, out: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import bar_with_ci

    regimes = out["regimes"]
    R = out["results"]
    labels = [TYPE_LABEL_KO[t] for t in ALL_TYPES]

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

    # (a) RE survival basin
    ax = fig.add_subplot(gs[0, 0])
    m = np.array([[R[rg]["surv_re"][:, i].mean() for i in range(len(ALL_TYPES))]
                  for rg in regimes]).mean(axis=0)
    ci = [boot_mean_ci(np.concatenate([R[rg]["surv_re"][:, i]
                                       for rg in regimes]))["ci"]
          for i in range(len(ALL_TYPES))]
    bar_with_ci(ax, labels, m, ci,
                colors=[TYPE_COLORS[t] for t in ALL_TYPES],
                ylabel="survival basin fraction", rotate=35)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"(a) RE survival basin (pooled regimes)\n"
                 f"threshold = {out['threshold']:.4f} (1/30)")

    # (b) ORE survival basin
    ax = fig.add_subplot(gs[0, 1])
    m = np.array([[R[rg]["surv_ore"][:, i].mean()
                   for i in range(len(ALL_TYPES))] for rg in regimes]).mean(axis=0)
    ci = [boot_mean_ci(np.concatenate([R[rg]["surv_ore"][:, i]
                                       for rg in regimes]))["ci"]
          for i in range(len(ALL_TYPES))]
    bar_with_ci(ax, labels, m, ci,
                colors=[TYPE_COLORS[t] for t in ALL_TYPES],
                ylabel="survival basin fraction", rotate=35)
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) ORE survival basin (pooled regimes)")

    # (c) mean terminal frequency (ORE)
    ax = fig.add_subplot(gs[0, 2])
    m = np.array([[R[rg]["freq_ore"][:, i].mean()
                   for i in range(len(ALL_TYPES))] for rg in regimes]).mean(axis=0)
    ci = [boot_mean_ci(np.concatenate([R[rg]["freq_ore"][:, i]
                                       for rg in regimes]))["ci"]
          for i in range(len(ALL_TYPES))]
    bar_with_ci(ax, labels, m, ci,
                colors=[TYPE_COLORS[t] for t in ALL_TYPES],
                ylabel="mean terminal frequency", rotate=35)
    ax.axhline(1.0 / len(ALL_TYPES), color="crimson", ls="--", lw=1.1,
               label="uniform initial (1/6)")
    ax.set_title("(c) ORE terminal composition"); ax.legend(fontsize=7)

    # (d) HalloReg survival by regime (RE vs ORE)
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(regimes)); w = 0.36
    for off, key, col, lab in ((-w / 2, "surv_re", "#4C72B0", "RE"),
                               (+w / 2, "surv_ore", "#C44E52", "ORE")):
        vals = [R[rg][key][:, HR].mean() for rg in regimes]
        errs = [np.array(boot_mean_ci(R[rg][key][:, HR])["ci"]) for rg in regimes]
        lo = [v - e[0] for v, e in zip(vals, errs)]
        hi = [e[1] - v for v, e in zip(vals, errs)]
        ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=3, color=col,
               label=lab, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL_KO[r] for r in regimes], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("HalloReg survival basin"); ax.set_ylim(0, 1.05)
    ax.set_title("(d) HalloReg survival by regime"); ax.legend(fontsize=7)

    # (e) representative RE trajectory (first regime)
    rg0 = regimes[0]
    ax = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(ALL_TYPES):
        ax.plot(R[rg0]["traj_re"][:, i], color=TYPE_COLORS[t],
                label=TYPE_LABEL_KO[t], lw=1.3)
    ax.set_xlabel("generation (integration step)")
    ax.set_ylabel("composition frequency")
    ax.set_title(f"(e) RE trajectory — {REGIME_LABEL_KO[rg0]}\n"
                 f"(uniform initial composition)")
    ax.legend(ncol=2, fontsize=6)

    # (f) representative ORE trajectory (first regime)
    ax = fig.add_subplot(gs[1, 2])
    for i, t in enumerate(ALL_TYPES):
        ax.plot(R[rg0]["traj_ore"][:, i], color=TYPE_COLORS[t],
                label=TYPE_LABEL_KO[t], lw=1.3)
    ax.set_xlabel("integration step (backward costate coupling)")
    ax.set_ylabel("composition frequency")
    ax.set_title(f"(f) ORE trajectory — {REGIME_LABEL_KO[rg0]}")
    ax.legend(ncol=2, fontsize=6)

    # (g) regime x type ORE survival heatmap
    ax = fig.add_subplot(gs[2, 0])
    M = np.array([[R[rg]["surv_ore"][:, i].mean()
                   for i in range(len(ALL_TYPES))] for rg in regimes])
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([REGIME_LABEL_KO[r] for r in regimes], fontsize=7)
    for i in range(len(regimes)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if M[i, j] > 0.55 else "black")
    ax.set_title("(g) regime x type ORE survival"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045)

    # (h) dominance basin (ORE)
    ax = fig.add_subplot(gs[2, 1])
    m = np.array([[R[rg]["dom_ore"][:, i].mean()
                   for i in range(len(ALL_TYPES))] for rg in regimes]).mean(axis=0)
    ci = [boot_mean_ci(np.concatenate([R[rg]["dom_ore"][:, i]
                                       for rg in regimes]))["ci"]
          for i in range(len(ALL_TYPES))]
    bar_with_ci(ax, labels, m, ci,
                colors=[TYPE_COLORS[t] for t in ALL_TYPES],
                ylabel="dominance basin fraction", rotate=35)
    ax.set_title("(h) ORE dominance basin\n"
                 "(share with the largest terminal frequency)")

    # (i) terminal CC rate: RE vs ORE
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(len(regimes)); w = 0.36
    for off, key, col, lab in ((-w / 2, "cc_re", "#4C72B0", "RE"),
                               (+w / 2, "cc_ore", "#C44E52", "ORE")):
        vals = [R[rg][key].mean() for rg in regimes]
        errs = [np.array(boot_mean_ci(R[rg][key])["ci"]) for rg in regimes]
        lo = [v - e[0] for v, e in zip(vals, errs)]
        hi = [e[1] - v for v, e in zip(vals, errs)]
        ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=3, color=col,
               label=lab, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL_KO[r] for r in regimes], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("expected CC rate of the terminal composition")
    ax.set_title("(i) terminal behavioural cooperation rate")
    ax.legend(fontsize=7)

    fig.suptitle("H5 — survival under replicator (RE) and optimal "
                 "replicator (ORE) dynamics",
                 fontsize=12, y=0.985)
    save_fig(fig, cfg, "H5_evolution")
