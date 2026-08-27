"""
experiments.h3_h4_population
============================

**H3  — in a mixed population under a stationary payoff structure,
does HalloReg earn more than the fixed strategies?**
**H3A — under the same conditions, does it contribute significantly
more to raising the population's mutual cooperation rate?**
**H4  — the same as H3 under non-stationary payoffs.**
**H4A — the same as H3A under non-stationary payoffs.**

Population composition: {TFT, GTFT, WSLS, ALLC, ALLD, HalloReg}
summing to 30. "Simulate as many combinations as possible" is
realised by the exact decomposition in `ipd.population`, which
evaluates **every** composition (C(29,5) = 118,755 with at least one
of each type). That this is an identity rather than an approximation
is documented in population.py.

Index 1 — payoff (H3 / H4)
--------------------------
With mu_i(n) the per-round mean payoff of type i in composition n,
HalloReg's advantage is

    Delta_i(n) = mu_HalloReg(n) - mu_i(n),
    i in {TFT, GTFT, WSLS, ALLC, ALLD}

**Important — the interpretive trap of the ALLD comparison.** ALLD
earns well in cooperator-rich compositions, but by transferring
others' payoff to itself, not by robust performance. The comparison
against ALLD is therefore registered as **exploratory only**, and the
confirmatory tests cover the four cooperative strategies
(TFT/GTFT/WSLS/ALLC). This decision is fixed before seeing results.

    - Confirmatory: Delta_i > 0 for each of the four cooperative
      rivals. The unit of testing is the **seed**: compositions are
      not independent (all derive from the same Pi), so using them as
      replicates would shrink p-values artificially. Per seed, an
      independently estimated Pi_s yields the all-composition mean
      Delta-bar_i(s), and a sign-flip permutation test is applied to
      that seed vector.

Index 2 — cooperation contribution (H3A / H4A)
----------------------------------------------
"Contributing more to the population's mutual cooperation" is
operationalised as a single individual's **marginal contribution**:
the change in CC rate when one HalloReg individual in composition n
is replaced by type i,

    Delta^CC_i(n) = CC(n) - CC(n - e_HalloReg + e_i)

A positive value means having HalloReg in that slot raises population
cooperation more than type i would. The substitution design is used
because the simple correlation "compositions with more HalloReg have
higher CC" is vulnerable to size and composition confounds;
substitution holds the population at 30 and changes exactly one slot,
removing them.

    - Confirmatory: Delta^CC_i > 0 for each of the four cooperative
      rivals (seed-level test).
    - Exploratory: the ALLD substitution, and the dose-response slope
      (number of HalloReg vs CC rate) across composition space.

H4 / H4A — non-stationary payoffs
---------------------------------
The same procedure is repeated in each of five non-stationary regimes
(blocks / oscillate / aba / drift / shock); the pooled test across
regimes is confirmatory and the per-regime results are
exploratory.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import ALL_TYPES, TYPE_LABEL_KO
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, effect_size_paired, fmt_es, one_sample_perm, slope_boot,
)
from AIF_IPD.ipd.payoff_schedule import NONSTATIONARY_REGIMES, REGIME_LABEL_KO
from AIF_IPD.ipd.population import (
    default_type_specs, enumerate_compositions, estimate_pair_matrices,
    population_cc, run_round_robin, subsample_compositions,
    substitution_delta_cc, type_payoffs,
)
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json

LOGGER = get_logger("HalloReg.H3H4")

HR = ALL_TYPES.index("halloreg")
#: Confirmatory comparison set — the four cooperative strategies
#: (ALLD is exploratory only)
COOP_RIVALS = ("tft", "gtft", "wsls", "allc")
TOTAL_AGENTS = 30


# ======================================================= Core computation
def analyze_regime(pair: dict, comps: np.ndarray) -> dict:
    """
    Compute payoff and cooperation indices over all compositions from
    one regime's type-pair matrices.

    Each seed's Pi_s / CCm_s is used separately, so the returned
    arrays are (seeds,) vectors — making the seed the replicate unit
    of the statistical tests.
    """
    names = pair["names"]
    S = pair["Pi_raw"].shape[2]
    k = len(names)

    # (seeds,) — each seed's all-composition mean payoff gap
    d_pay = {n: np.zeros(S) for n in names}
    # (seeds,) — each seed's mean substitution CC contribution
    d_cc = {n: np.zeros(S) for n in names}
    # Dose-response slope (number of HalloReg -> CC rate)
    dose = np.zeros(S)

    for s in range(S):
        Pi = pair["Pi_raw"][:, :, s]
        CCm = pair["CCm_raw"][:, :, s]
        mu = type_payoffs(Pi, comps)                 # (M, k)
        for i, n in enumerate(names):
            if i == HR:
                continue
            d_pay[n][s] = float(np.mean(mu[:, HR] - mu[:, i]))
            d_cc[n][s] = float(np.nanmean(
                substitution_delta_cc(CCm, comps, HR, i)))
        cc = population_cc(CCm, comps)
        sl = np.polyfit(comps[:, HR].astype(float), cc, 1)[0]
        dose[s] = float(sl)

    # Composition-level indices at the representative (seed-mean Pi)
    # — for the figures
    mu_mean = type_payoffs(pair["Pi"], comps)
    cc_mean = population_cc(pair["CCm"], comps)

    return {"names": names, "delta_payoff": d_pay, "delta_cc": d_cc,
            "dose_slope": dose, "mu_by_comp": mu_mean, "cc_by_comp": cc_mean,
            "n_comps": int(len(comps))}


def _confirm_block(reg: Registry, hyp: str, tag: str,
                   deltas: Dict[str, np.ndarray], unit: str) -> None:
    """Register the confirmatory tests for the four cooperative
    rivals plus the exploratory ALLD test."""
    for rival in COOP_RIVALS:
        d = deltas[rival]
        t = one_sample_perm(d, 0.0, alternative="greater")
        e = effect_size_paired(d)
        ci = boot_mean_ci(d)
        reg.confirm(hyp, f"{tag} vs {TYPE_LABEL_KO[rival]}", t["p"],
                    direction_ok=bool(ci["mean"] > 0),
                    effect=f"Δ{unit}={ci['mean']:+.4f} "
                           f"[{ci['ci'][0]:+.4f}, {ci['ci'][1]:+.4f}], "
                           f"{fmt_es(e, 'dz')}")
    d = deltas["alld"]
    t = one_sample_perm(d, 0.0, alternative="greater")
    ci = boot_mean_ci(d)
    reg.explore(hyp, f"{tag} vs ALLD (interpret with care: payoff "
                     f"transfer through exploitation)", t["p"],
                effect=f"Δ{unit}={ci['mean']:+.4f}")


# =================================================================== Run
def _prepare_comps(cfg: Config) -> np.ndarray:
    """
    Enumerate compositions, requiring at least one of each type: with
    zero individuals of type i, neither mu_i nor the substitution
    contrast is defined.
    """
    comps = enumerate_compositions(TOTAL_AGENTS, len(ALL_TYPES), min_each=1)
    if cfg.max_compositions and len(comps) > cfg.max_compositions:
        LOGGER.info("  composition subsampling: %d -> %d",
                    len(comps), cfg.max_compositions)
        comps = subsample_compositions(comps, cfg.max_compositions, seed=7)
    return comps


def run_stationary(cfg: Config, reg: Registry, comps: np.ndarray) -> dict:
    """H3 / H3A — stationary payoff structure."""
    LOGGER.info("[H3/H3A] stationary payoffs — %d compositions, "
                "%d type pairs estimated",
                len(comps), len(ALL_TYPES) * (len(ALL_TYPES) + 1) // 2)
    specs = default_type_specs(cfg.halloreg_kwargs())
    pair = estimate_pair_matrices(specs, cfg.rounds, cfg.seeds, cfg.jobs,
                                  eval_from=getattr(cfg, "eval_from", 0),
                                  regime=None, env_error=cfg.env_error,
                                  seed_offset=0)
    an = analyze_regime(pair, comps)

    _confirm_block(reg, "H3", "payoff", an["delta_payoff"], "payoff")
    _confirm_block(reg, "H3A", "marginal CC contribution", an["delta_cc"],
                   "CC")

    # Exploratory: dose-response slope
    t = one_sample_perm(an["dose_slope"], 0.0, alternative="greater")
    ci = boot_mean_ci(an["dose_slope"])
    reg.explore("H3A", "dose-response: slope of population CC rate on "
                "the number of HalloReg > 0", t["p"],
                effect=f"slope={ci['mean']:+.5f}/individual")

    # ---- Direct verification of the analytic decomposition ----
    check = _verify_decomposition(cfg, specs, pair, comps, regime=None)
    reg.explore("H3", "analytic decomposition vs direct round robin "
                "(verification)",
                1.0, effect=f"max |error| CC={check['max_abs_cc_err']:.4f}, "
                            f"payoff={check['max_abs_pay_err']:.4f}")

    return {"pair": pair, "analysis": an, "verify": check}


def run_nonstationary(cfg: Config, reg: Registry, comps: np.ndarray) -> dict:
    """H4 / H4A — non-stationary payoff structures (5 regimes)."""
    regimes = NONSTATIONARY_REGIMES
    specs = default_type_specs(cfg.halloreg_kwargs())
    per_regime = {}

    for ri, rg in enumerate(regimes):
        LOGGER.info("[H4/H4A] regime '%s' (%d/%d)", rg, ri + 1,
                    len(regimes))
        pair = estimate_pair_matrices(specs, cfg.rounds, cfg.seeds, cfg.jobs,
                                  eval_from=getattr(cfg, "eval_from", 0),
                                      regime=rg, env_error=cfg.env_error,
                                      seed_offset=1000 * (ri + 1))
        per_regime[rg] = {"pair": pair, "analysis": analyze_regime(pair, comps)}

    # ---- Confirmatory: pooled across regimes (seed x regime as
    # replicates) ----
    pooled_pay = {n: np.concatenate(
        [per_regime[rg]["analysis"]["delta_payoff"][n] for rg in regimes])
        for n in ALL_TYPES}
    pooled_cc = {n: np.concatenate(
        [per_regime[rg]["analysis"]["delta_cc"][n] for rg in regimes])
        for n in ALL_TYPES}

    _confirm_block(reg, "H4", "payoff (non-stationary, pooled)",
                   pooled_pay, "payoff")
    _confirm_block(reg, "H4A", "marginal CC contribution "
                   "(non-stationary, pooled)", pooled_cc, "CC")

    # ---- Exploratory: per regime ----
    for rg in regimes:
        an = per_regime[rg]["analysis"]
        for rival in COOP_RIVALS:
            t = one_sample_perm(an["delta_payoff"][rival], 0.0,
                                alternative="greater")
            ci = boot_mean_ci(an["delta_payoff"][rival])
            reg.explore("H4", f"[{REGIME_LABEL_KO[rg]}] payoff vs "
                              f"{TYPE_LABEL_KO[rival]}", t["p"],
                        effect=f"Δ={ci['mean']:+.4f}")
        t = one_sample_perm(an["dose_slope"], 0.0, alternative="greater")
        ci = boot_mean_ci(an["dose_slope"])
        reg.explore("H4A", f"[{REGIME_LABEL_KO[rg]}] dose-response slope",
                    t["p"],
                    effect=f"slope={ci['mean']:+.5f}/individual")

    return {"regimes": regimes, "per_regime": per_regime,
            "pooled_payoff": pooled_pay, "pooled_cc": pooled_cc}


def _verify_decomposition(cfg: Config, specs: dict, pair: dict,
                          comps: np.ndarray, regime: Optional[str],
                          n_check: int = 3) -> dict:
    """
    Verify directly that the analytic decomposition matches an actual
    round robin.

    For a few randomly drawn compositions, run the full 30-agent round
    robin (435 dyads) and compare against the decomposition's
    prediction. This is expensive, so composition count and seeds are
    kept minimal.
    """
    rng = np.random.default_rng(3)
    idx = rng.choice(len(comps), size=min(n_check, len(comps)), replace=False)
    rows = []
    for j, i in enumerate(idx):
        c = comps[i]
        direct = run_round_robin(c, specs, cfg.rounds, seed=100 + j,
                                 eval_from=getattr(cfg, "eval_from", 0),
                                 regime=regime, env_error=cfg.env_error,
                                 persistent=False, n_jobs=cfg.jobs)
        pred_cc = float(population_cc(pair["CCm"], c))
        pred_mu = type_payoffs(pair["Pi"], c)
        rows.append({
            "counts": c.tolist(),
            "cc_direct": direct["cc_rate"], "cc_analytic": pred_cc,
            "cc_err": direct["cc_rate"] - pred_cc,
            "payoff_direct": direct["by_type_payoff"],
            "payoff_analytic": {n: float(pred_mu[i2])
                                for i2, n in enumerate(pair["names"])},
        })
        LOGGER.info("  check %d: CC direct=%.4f analytic=%.4f "
                    "(diff=%+.4f)",
                    j + 1, direct["cc_rate"], pred_cc,
                    direct["cc_rate"] - pred_cc)
    max_cc = max(abs(r["cc_err"]) for r in rows)
    max_pay = max(
        max(abs(r["payoff_direct"][n] - r["payoff_analytic"][n])
            for n in r["payoff_analytic"])
        for r in rows)
    return {"rows": rows, "max_abs_cc_err": max_cc, "max_abs_pay_err": max_pay}


def run(cfg: Config, reg: Registry) -> dict:
    comps = _prepare_comps(cfg)
    LOGGER.info("[H3-H4A] composition space size = %d "
                "(%d agents, at least one per type)",
                len(comps), TOTAL_AGENTS)
    stat = run_stationary(cfg, reg, comps)
    nons = run_nonstationary(cfg, reg, comps)
    out = {"n_compositions": int(len(comps)),
           "stationary": stat, "nonstationary": nons}
    _plot_stationary(cfg, stat, comps)
    _plot_nonstationary(cfg, nons)
    save_json({"n_compositions": int(len(comps)),
               "stationary": {"analysis": _summarize(stat["analysis"]),
                              "Pi": stat["pair"]["Pi"],
                              "CCm": stat["pair"]["CCm"],
                              "verify": stat["verify"]},
               "nonstationary": {
                   "regimes": nons["regimes"],
                   "per_regime": {rg: {
                       "Pi": nons["per_regime"][rg]["pair"]["Pi"],
                       "CCm": nons["per_regime"][rg]["pair"]["CCm"],
                       "analysis": _summarize(
                           nons["per_regime"][rg]["analysis"])}
                       for rg in nons["regimes"]}}},
              cfg.results / "H3_H4.json")
    return out


def _summarize(an: dict) -> dict:
    """
    Summary for JSON serialisation.

    `mu_by_comp` / `cc_by_comp` have **one row per composition**
    (118,755 under full enumeration). Storing them verbatim would make
    each regime hundreds of megabytes and render the result file
    useless, so composition-level raw data is used only in the figures
    and only distributional summaries are written to JSON. The
    seed-level vectors (delta_payoff / delta_cc / dose_slope) are the
    replicate units of the statistical tests and are preserved as
    is.
    """
    out = {k: v for k, v in an.items()
           if k not in ("mu_by_comp", "cc_by_comp")}
    mu = an["mu_by_comp"]
    cc = an["cc_by_comp"]
    out["mu_by_comp_summary"] = {
        name: {"mean": float(mu[:, i].mean()),
               "sd": float(mu[:, i].std(ddof=1)),
               "q05": float(np.percentile(mu[:, i], 5)),
               "q50": float(np.percentile(mu[:, i], 50)),
               "q95": float(np.percentile(mu[:, i], 95))}
        for i, name in enumerate(an["names"])}
    out["cc_by_comp_summary"] = {
        "mean": float(cc.mean()), "sd": float(cc.std(ddof=1)),
        "q05": float(np.percentile(cc, 5)), "q50": float(np.percentile(cc, 50)),
        "q95": float(np.percentile(cc, 95))}
    return out


# ============================================================== Figures
def _plot_stationary(cfg: Config, res: dict, comps: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from .common import annotate_n, bar_with_ci

    pair, an = res["pair"], res["analysis"]
    labels = [TYPE_LABEL_KO[t] for t in ALL_TYPES]
    rivals = [t for t in ALL_TYPES if t != "halloreg"]
    rlabels = [TYPE_LABEL_KO[t] for t in rivals]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

    # (a) payoff matrix Pi
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(pair["Pi"], cmap="viridis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("(a) type-pair payoff matrix Pi\n"
                 "(per-round payoff of the row type)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{pair['Pi'][i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (b) mutual cooperation matrix CCm
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(pair["CCm"], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("(b) type-pair mutual cooperation rate CCm")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = pair["CCm"][i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.55 else "black")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (c) H3 — payoff advantage
    ax = fig.add_subplot(gs[0, 2])
    means = [an["delta_payoff"][r].mean() for r in rivals]
    cis = [boot_mean_ci(an["delta_payoff"][r])["ci"] for r in rivals]
    cols = ["#4C72B0" if r in COOP_RIVALS else "#937860" for r in rivals]
    bar_with_ci(ax, rlabels, means, cis, colors=cols,
                ylabel="delta per-round payoff", rotate=30)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title(f"(c) H3 — HalloReg minus rival payoff\n"
                 f"(mean over {an['n_comps']:,} compositions, "
                 f"n={cfg.seeds} seeds)")

    # (d) H3A — marginal CC contribution
    ax = fig.add_subplot(gs[1, 0])
    means = [an["delta_cc"][r].mean() for r in rivals]
    cis = [boot_mean_ci(an["delta_cc"][r])["ci"] for r in rivals]
    bar_with_ci(ax, rlabels, means, cis, colors=cols,
                ylabel="delta population CC rate (per substitution)",
                rotate=30)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title("(d) H3A — substitution contribution\n"
                 "CC(n) - CC(one HalloReg replaced by the rival)")

    # (e) dose-response: number of HalloReg vs CC rate
    ax = fig.add_subplot(gs[1, 1])
    nh = comps[:, HR]
    cc = an["cc_by_comp"]
    binned = [cc[nh == v] for v in range(1, min(int(nh.max()), 25) + 1)]
    xs = [v for v in range(1, min(int(nh.max()), 25) + 1) if len(binned[v - 1])]
    ax.boxplot([binned[v - 1] for v in xs], positions=xs, widths=0.6,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#DA8BC3", alpha=0.7))
    ax.set_xlabel("number of HalloReg in the population")
    ax.set_ylabel("population CC rate")
    ax.set_title(f"(e) dose-response (slope "
                 f"{an['dose_slope'].mean():+.5f}/individual)")
    ax.set_xticks(xs[::4]); ax.set_xticklabels([str(v) for v in xs[::4]])

    # (f) distribution of per-type mean payoff across compositions
    ax = fig.add_subplot(gs[1, 2])
    data = [an["mu_by_comp"][:, i] for i in range(len(ALL_TYPES))]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    showfliers=False)
    for patch, t in zip(bp["boxes"], ALL_TYPES):
        patch.set_facecolor(TYPE_COLORS[t]); patch.set_alpha(0.8)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("mean payoff per round")
    ax.set_title("(f) payoff distribution by type across all "
                 "compositions")

    fig.suptitle("H3 / H3A — mixed population under stationary payoffs "
                 "(30 agents, all compositions evaluated analytically)",
                 fontsize=12, y=0.98)
    save_fig(fig, cfg, "H3_stationary_population")


def _plot_nonstationary(cfg: Config, res: dict) -> None:
    import matplotlib.pyplot as plt
    from AIF_IPD.ipd.payoff_schedule import regime_trace
    from .common import bar_with_ci

    regimes = res["regimes"]
    rivals = [t for t in ALL_TYPES if t != "halloreg"]
    rlabels = [TYPE_LABEL_KO[t] for t in rivals]
    cols = ["#4C72B0" if r in COOP_RIVALS else "#937860" for r in rivals]

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32,
                          height_ratios=[0.75, 1, 1])

    # (a) CI trajectory per regime
    ax = fig.add_subplot(gs[0, :])
    for rg in regimes:
        ax.plot(regime_trace(rg, cfg.rounds), lw=1.3, label=REGIME_LABEL_KO[rg])
    ax.axhline(0.4, color="#888888", ls=":", lw=1,
               label="default PD (CI=0.4)")
    ax.axhline(0.0, color="crimson", ls="--", lw=0.9)
    ax.set_xlabel("round"); ax.set_ylabel("cooperation index CI")
    ax.set_title("(a) non-stationary payoff regimes — "
                 "CI = (R-P)/(T-S) trajectories\n"
                 "CI<0 deadlock | CI=0.4 default PD | CI>=1 harmony")
    ax.legend(ncol=3, fontsize=7)

    # (b) H4 — pooled payoff advantage
    ax = fig.add_subplot(gs[1, 0])
    means = [res["pooled_payoff"][r].mean() for r in rivals]
    cis = [boot_mean_ci(res["pooled_payoff"][r])["ci"] for r in rivals]
    bar_with_ci(ax, rlabels, means, cis, colors=cols,
                ylabel="delta per-round payoff", rotate=30)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title("(b) H4 — pooled payoff advantage, non-stationary")

    # (c) H4A — pooled CC contribution
    ax = fig.add_subplot(gs[1, 1])
    means = [res["pooled_cc"][r].mean() for r in rivals]
    cis = [boot_mean_ci(res["pooled_cc"][r])["ci"] for r in rivals]
    bar_with_ci(ax, rlabels, means, cis, colors=cols,
                ylabel="delta population CC rate", rotate=30)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title("(c) H4A — pooled marginal CC contribution, "
                 "non-stationary")

    # (d) regime x rival payoff-advantage heatmap
    ax = fig.add_subplot(gs[1, 2])
    M = np.array([[res["per_regime"][rg]["analysis"]["delta_payoff"][r].mean()
                   for r in rivals] for rg in regimes])
    vmax = np.max(np.abs(M)) if np.max(np.abs(M)) > 0 else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(rivals))); ax.set_xticklabels(rlabels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([REGIME_LABEL_KO[r] for r in regimes], fontsize=7)
    for i in range(len(regimes)):
        for j in range(len(rivals)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6.5)
    ax.set_title("(d) payoff advantage by regime")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (e) regime x rival CC-contribution heatmap
    ax = fig.add_subplot(gs[2, 0])
    M = np.array([[res["per_regime"][rg]["analysis"]["delta_cc"][r].mean()
                   for r in rivals] for rg in regimes])
    vmax = np.max(np.abs(M)) if np.max(np.abs(M)) > 0 else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(rivals))); ax.set_xticklabels(rlabels, rotation=45,
                                                          ha="right")
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([REGIME_LABEL_KO[r] for r in regimes], fontsize=7)
    for i in range(len(regimes)):
        for j in range(len(rivals)):
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                    fontsize=6)
    ax.set_title("(e) marginal CC contribution by regime")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045)

    # (f) dose-response slope by regime
    ax = fig.add_subplot(gs[2, 1])
    means = [res["per_regime"][rg]["analysis"]["dose_slope"].mean()
             for rg in regimes]
    cis = [boot_mean_ci(res["per_regime"][rg]["analysis"]["dose_slope"])["ci"]
           for rg in regimes]
    bar_with_ci(ax, [REGIME_LABEL_KO[r] for r in regimes], means, cis,
                colors="#55A868", ylabel="CC-rate slope (/individual)",
                rotate=35)
    ax.axhline(0, color="crimson", ls="--", lw=1.1)
    ax.set_title("(f) dose-response slope by regime")

    # (g) per-type payoff by regime (mean over compositions)
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(len(regimes))
    w = 0.13
    for i, t in enumerate(ALL_TYPES):
        vals = [res["per_regime"][rg]["analysis"]["mu_by_comp"][:, i].mean()
                for rg in regimes]
        ax.bar(x + (i - 2.5) * w, vals, w, color=TYPE_COLORS[t],
               label=TYPE_LABEL_KO[t], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL_KO[r] for r in regimes], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("mean payoff per round")
    ax.set_title("(g) regime x type mean payoff")
    ax.legend(ncol=2, fontsize=6)

    fig.suptitle("H4 / H4A — mixed population under non-stationary "
                 "payoffs", fontsize=12, y=0.985)
    save_fig(fig, cfg, "H4_nonstationary_population")
