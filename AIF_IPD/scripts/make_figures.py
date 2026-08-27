"""
make_figures — paper figures 1-6 from the experiment results (v3.9.7)
====================================================================

Reads a results directory produced by ``run_ipd_experiment.py`` and
renders the six paper figures. There is no separate simulation pass:
every figure is a view on data the hypothesis tests already produced,
so the figures and the reported statistics come from the same runs at
the same number of seeds.

    python -B AIF_IPD/scripts/run_ipd_experiment.py --seeds 120 \\
        --rounds 800 --eval-from 600 --particles 600 --jobs 16 \\
        --results results_v396
    python -B AIF_IPD/scripts/make_figures.py --results results_v396

Sources
-------
    fig1  lambda separation by partner          ARCH.lam_by_partner
          exploiter vs noisy-TFT discrimination H2.discrimination
    fig2  intent tracking across switches       H1A.tracking
    fig3  stationary 6x6 payoff matrix          H3_H4.stationary.Pi
    fig4  non-stationary payoff schedules       payoff_schedule (analytic)
    fig5  6x6 matrix per regime + advantage     H3_H4.nonstationary
    fig6  substitution effect on group CC       H3_H4 *.analysis.delta_cc

Each figure carries the seed and round count from SUMMARY.json in its
caption, so a rendered figure always states its own scale. A figure
whose source file is missing is skipped with a note rather than drawn
from a different run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from AIF_IPD.experiments.common import TYPE_COLORS  # noqa: E402
from AIF_IPD.ipd.payoff_schedule import (  # noqa: E402
    NONSTATIONARY_REGIMES, get_regime,
)

NAME_EN = {"halloreg": "HalloReg", "tft": "TFT", "gtft": "GTFT",
           "wsls": "WSLS", "allc": "ALLC", "alld": "ALLD",
           "blocks": "Blocks", "oscillate": "Oscillate", "aba": "ABA",
           "drift": "Drift", "shock": "Shock", "stationary": "Stationary",
           "recip_expl_recon": "Reciprocator / exploiter / reconciliation",
           "coop_trap": "Cooperator / trap", "wsls_flip": "WSLS flip"}
# One palette across every figure: the per-type colours are the same
# TYPE_COLORS the experiment modules use, so a strategy keeps its
# colour whether it appears in a paper figure or a diagnostic figure.
# Series that are not a strategy (an inferred quantity, a ground-truth
# reference) take the two neutral entries of the same palette.
LAM_SELF = TYPE_COLORS["halloreg"]          # the regulator's own lambda
LAM_PARTNER = TYPE_COLORS["empathic_hi"]    # inferred partner lambda
TRUTH = "#333333"                           # ground truth reference
# Sequential map for payoff levels, diverging map for signed contrasts.
CMAP_LEVEL, CMAP_DIFF = "viridis", "RdBu_r"
# HalloReg last, so the learning agent sits in the bottom row and
# the right-hand column of every matrix.
ORDER6 = ("tft", "gtft", "wsls", "allc", "alld", "halloreg")


# ------------------------------------------------------------- utilities
def _load(results: str, name: str) -> Optional[dict]:
    path = os.path.join(results, f"{name}.json")
    if not os.path.exists(path):
        print(f"[skip] {name}.json not found in {results}")
        return None
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


def _scale(summary: Optional[dict]) -> str:
    if not summary:
        return ""
    c = summary.get("config", {})
    return (f"n = {c.get('seeds', '?')} seeds, {c.get('rounds', '?')} rounds, "
            f"{c.get('particles', '?')} particles")


def _save(fig, results: str, name: str, caption: str) -> None:
    if caption:
        fig.text(0.995, 0.005, caption, ha="right", va="bottom",
                 fontsize=7, color="#555555")
    d = os.path.join(results, "figures_paper")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {path}")


def _heat(ax, M, xt, yt, cmap, title, fmt="{:.2f}", center=None):
    M = np.asarray(M, dtype=float)
    kw = {}
    if center is not None:
        v = float(np.nanmax(np.abs(M - center)))
        kw = {"vmin": center - v, "vmax": center + v}
    im = ax.imshow(M, cmap=cmap, aspect="auto", **kw)
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels(xt, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(yt)))
    ax.set_yticklabels(yt, fontsize=8)
    # Cell labels are always black, as in the pre-v3.9.7 figures. A
    # luminance-dependent colour was tried and made the tables harder
    # to read, because the switch point falls in the middle of the
    # value range rather than at a visual boundary.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center",
                        va="center", fontsize=7.5, color="black")
    ax.set_title(title, fontsize=12)
    return im


def _band(ax, mean, lo, hi, color, label):
    x = np.arange(len(mean))
    ax.plot(x, mean, color=color, lw=1.5, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)


def _reorder(M, agents):
    idx = [agents.index(a) for a in ORDER6 if a in agents]
    return (np.asarray(M, dtype=float)[np.ix_(idx, idx)],
            [NAME_EN.get(agents[i], agents[i]) for i in idx])


# ------------------------------------------------------------------ fig1
def fig1(results, summary):
    arch, h2 = _load(results, "ARCH"), _load(results, "H2")
    has_arch = arch is not None and "lam_by_partner" in arch
    has_h2 = bool(h2) and "discrimination" in h2
    if not has_arch:
        # Results written before v3.9.7 have no lambda bands in
        # ARCH.json. Panel (b) still renders from H2; panel (a) needs
        # the ARCH experiment re-run with v3.9.7 or later.
        print("[note] fig1 panel (a) skipped: ARCH.json has no "
              "lam_by_partner (results predate v3.9.7)")
    if not (has_arch or has_h2):
        print("[skip] fig1 needs ARCH.json or H2.json")
        return
    if has_arch and has_h2:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    else:
        fig, ax0 = plt.subplots(figsize=(6.2, 4.4))
        a1 = a2 = ax0
    if has_arch:
        for k in arch.get("probe_types", list(arch["lam_by_partner"])):
            b = arch["lam_by_partner"][k]
            _band(a1, b["mean"], b["lo"], b["hi"],
                  TYPE_COLORS.get(k, "#8C8C8C"), NAME_EN.get(k, k))
        a1.set_xlabel("Round")
        a1.set_ylabel(r"$\lambda_t$")
        a1.set_ylim(-0.02, 1.02)
        a1.legend(fontsize=8, ncol=2)
        a1.set_title((r"(a) Endogenous $\lambda$ by opponent type"
                      if has_h2 else
                      r"Endogenous $\lambda$ by opponent type")
                     + "\n(mean, IQR band)", fontsize=12)

    if has_h2:
        d = h2["discrimination"]
        for k, c, lab in (("noisy_tft", TYPE_COLORS["tft"], "Noisy TFT"),
                          ("exploiter", TYPE_COLORS["alld"], "Exploiter")):
            tr = np.asarray(d["lam_traces"][k], dtype=float)
            _band(a2, tr.mean(0), np.percentile(tr, 25, 0),
                  np.percentile(tr, 75, 0), c, lab)
        a2.set_xlabel("Round")
        a2.set_ylabel(r"$\lambda_t$")
        a2.set_ylim(-0.02, 1.02)
        a2.legend(fontsize=8)
        acc = float(d.get("accuracy", float("nan")))
        a2.set_title(("(b) " if has_arch else "")
                     + "Behavioral level discrimination",
                     fontsize=12)
    fig.tight_layout()
    _save(fig, results, "fig1_lambda_dynamics", _scale(summary))


# ------------------------------------------------------------------ fig2
def fig2(results, summary):
    """Lambda dynamics across intent switches.

    Preferred source: H1A.tracking.lam_bands (v3.9.7+), the regulator's
    own lambda and the inferred partner lambda. Results written before
    v3.9.7 do not carry it; the panel then falls back to the tracking
    signal itself (inferred vs true cooperativeness), which is what the
    same file does contain.
    """
    d = _load(results, "H1A")
    if d is None:
        return
    tr = d["tracking"]
    scen = tr["scenarios"]
    bands = tr.get("lam_bands")
    fig, axes = plt.subplots(1, len(scen), figsize=(4.4 * len(scen), 3.9),
                             squeeze=False, sharey=True)
    for ax, sc in zip(axes[0], scen):
        truth = np.asarray(tr["truth"][sc], dtype=float)
        if bands and sc in bands:
            for key, c, lab in (("lam", LAM_SELF, r"$\lambda_t$ (self)"),
                                ("lam_j", LAM_PARTNER,
                                 r"$\hat{\lambda}_j$ (partner)")):
                b = bands[sc][key]
                _band(ax, b["mean"], b["lo"], b["hi"], c, lab)
            ax.plot(np.arange(len(truth)), truth, color=TRUTH, lw=1.2,
                    ls="--", label="Partner cooperativeness")
            ylab = r"$\lambda$"
        else:
            traces = np.asarray(tr["traces"][sc], dtype=float)
            _band(ax, traces.mean(0), np.percentile(traces, 25, 0),
                  np.percentile(traces, 75, 0), LAM_PARTNER,
                  "Inferred (ToM)")
            ax.plot(np.arange(len(truth)), truth, color=TRUTH, lw=1.4,
                    ls="--", label="True cooperativeness")
            ylab = "P(cooperate)"
        for sw in tr.get("switch_rounds", []):
            ax.axvline(sw, color="#999999", lw=0.8, ls=":")
        r = float(np.nanmean(tr["corr"][sc]))
        ax.set_title(f"{NAME_EN.get(sc, sc)}",
                     fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel(ylab)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    caption = _scale(summary)
    if not bands:
        caption += "  [pre-v3.9.7 results: lambda bands unavailable]"
    _save(fig, results, "fig2_trait_switch", caption)


# ------------------------------------------------------------------ fig3
def fig3(results, summary):
    d = _load(results, "H3_H4")
    if d is None:
        return
    agents = d["stationary"]["analysis"]["names"]
    M, names = _reorder(d["stationary"]["Pi"], agents)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = _heat(ax, M, names, names, CMAP_LEVEL,
               "Expected payoff per round, stationary payoffs")
    ax.grid(False)
    ax.set_xlabel("Partner")
    ax.set_ylabel("Focal agent")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    _save(fig, results, "fig3_payoff_matrix_stationary", _scale(summary))


# ------------------------------------------------------------------ fig4
def fig4(results, summary):
    rounds = 800
    if summary:
        rounds = int(summary.get("config", {}).get("rounds", rounds))
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    reg_colors = [TYPE_COLORS[k] for k in
                  ("tft", "gtft", "wsls", "allc", "alld")]
    for r, c in zip(NONSTATIONARY_REGIMES, reg_colors):
        f = get_regime(r)
        ax.plot(range(rounds), [f(t, rounds) for t in range(rounds)],
                lw=1.4, color=c, label=NAME_EN.get(r, r))
    ax.grid(False)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cooperation index")
    ax.legend(fontsize=8, ncol=5, loc="upper right", framealpha=0.9)
    ax.set_title("Non-stationary payoff schedules", fontsize=12)
    ax.margins(x=0.01)
    fig.tight_layout()
    _save(fig, results, "fig4_payoff_schedules",
          f"analytic schedules, {rounds} rounds")


# ------------------------------------------------------------------ fig5
def fig5(results, summary):
    d = _load(results, "H3_H4")
    if d is None:
        return
    ns = d["nonstationary"]
    regs = list(ns["regimes"])
    agents = d["stationary"]["analysis"]["names"]
    Ms = {r: np.asarray(ns["per_regime"][r]["Pi"], dtype=float) for r in regs}
    avg, names = _reorder(np.nanmean(np.stack([Ms[r] for r in regs]), axis=0),
                          agents)
    hr = agents.index("halloreg")
    fx = [a for a in ORDER6 if a in agents and a != "halloreg"]
    D = np.array([[np.nanmean([Ms[r][hr, agents.index(j)]
                               - Ms[r][agents.index(s), agents.index(j)]
                               for j in fx]) for s in fx] for r in regs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.9))
    im1 = _heat(a1, avg, names, names, CMAP_LEVEL,
                "Expected payoff per round,\nnon-stationary regimes")
    a1.grid(False)
    a1.set_xlabel("Partner")
    a1.set_ylabel("Focal agent")
    fig.colorbar(im1, ax=a1, shrink=0.85)
    im2 = _heat(a2, D, [NAME_EN[a] for a in fx],
                [NAME_EN.get(r, r) for r in regs], CMAP_DIFF,
                "Payoff advantage of HalloReg over each\nfixed strategy, "
                "averaged over partners", fmt="{:+.2f}", center=0.0)
    a2.grid(False)
    a2.set_xlabel("Fixed strategy")
    a2.set_ylabel("Payoff regime")
    fig.colorbar(im2, ax=a2, shrink=0.85)
    fig.tight_layout()
    _save(fig, results, "fig5_payoff_nonstationary", _scale(summary))


# ------------------------------------------------------------------ fig6
def fig6(results, summary):
    d = _load(results, "H3_H4")
    if d is None:
        return
    regs = ["stationary"] + list(d["nonstationary"]["regimes"])
    an = {"stationary": d["stationary"]["analysis"]}
    an.update({r: d["nonstationary"]["per_regime"][r]["analysis"]
               for r in d["nonstationary"]["regimes"]})
    replaced = [a for a in ORDER6
                if a != "halloreg" and a in an["stationary"]["delta_cc"]]
    M = np.array([[np.nanmean(an[r]["delta_cc"][x]) for x in replaced]
                  for r in regs])
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    im = _heat(ax, M, [NAME_EN[a] for a in replaced],
               [NAME_EN.get(r, r) for r in regs], CMAP_DIFF,
               "Change in group mutual cooperation when one individual\n"
               "of each type is replaced by HalloReg",
               fmt="{:+.3f}", center=0.0)
    ax.grid(False)
    ax.set_xlabel("Replaced strategy")
    ax.set_ylabel("Payoff regime")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    n_comps = an["stationary"].get("n_comps", "?")
    _save(fig, results, "fig6_substitution_effect",
          f"{_scale(summary)}; {n_comps} compositions")


FIGS = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render paper figures 1-6 from an experiment results "
                    "directory. No simulation is run.")
    p.add_argument("--results", default="results",
                   help="directory written by run_ipd_experiment.py")
    p.add_argument("--figs", default="1,2,3,4,5,6")
    args = p.parse_args()
    summary = _load(args.results, "SUMMARY")
    for f in [s.strip() for s in args.figs.split(",") if s.strip()]:
        if f not in FIGS:
            print(f"[skip] unknown figure '{f}'")
            continue
        FIGS[f](args.results, summary)


if __name__ == "__main__":
    main()
