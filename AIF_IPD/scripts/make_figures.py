#!/usr/bin/env python3
"""
make_figures — render the poster figures from saved figdata (v3.9.0)
=====================================================================
Reads only the <out>/figdata/*.json files left by run_figure_sims.py
and draws them (fully separated from simulation).
Output: <out>/figures_en/fig{1..6}_*.png

    python3 -B AIF_IPD/scripts/make_figures.py --out results_figs
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAME_EN = {"halloreg": "HalloReg", "tft": "TFT", "gtft": "GTFT",
           "wsls": "WSLS", "allc": "ALLC", "alld": "ALLD",
           "stationary": "Stationary", "blocks": "Blocks",
           "oscillate": "Oscillate", "aba": "ABA", "drift": "Drift",
           "shock": "Shock"}
SCEN_EN = {"recip_expl_recon": "Reciprocity → Exploit → Reconcile",
           "coop_trap": "Cooperation trap",
           "wsls_flip": "WSLS flip"}
OPP_COLORS = {"allc": "#4C72B0", "gtft": "#55A868", "tft": "#C44E52",
              "wsls": "#8172B2", "alld": "#937860"}


DISPLAY_ORDER = ("tft", "gtft", "wsls", "allc", "alld", "halloreg")


def _reorder6(M, agents):
    """Reorder a 6x6 matrix (rows and columns alike) into DISPLAY_ORDER
    so both axes carry the same, diagonal-symmetric layout."""
    import numpy as _np
    idx = [agents.index(a) for a in DISPLAY_ORDER]
    return _np.asarray(M, float)[_np.ix_(idx, idx)], \
        [NAME_EN[a] for a in DISPLAY_ORDER]


def _ld(out, name):
    path = os.path.join(out, "figdata", f"{name}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def _heat(ax, M, xt, yt, cmap, title, fmt="{:.2f}", center=None):
    M = np.asarray(M, float)
    if center is not None:
        v = np.nanmax(np.abs(M - center)) or 1.0
        im = ax.imshow(M, cmap=cmap, vmin=center - v, vmax=center + v)
    else:
        im = ax.imshow(M, cmap=cmap)
    ax.set_xticks(range(len(xt)), xt, rotation=45, ha="right")
    ax.set_yticks(range(len(yt)), yt)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=7)
    return im


def fig1(out):
    d = _ld(out, "fig1")
    if d is None:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for k, tr in d["opp"].items():
        a1.plot(tr, lw=1.4, color=OPP_COLORS.get(k), label=NAME_EN[k])
    a1.set(xlabel="Round", ylabel=r"Empathy weight $\lambda$",
           title=r"(a) Allostatic $\lambda$ by opponent type")
    a1.legend(fontsize=8, ncol=2)
    a1.set_ylim(-0.05, 1.05)
    a2.plot(d["disc"]["noisy_tft"], lw=1.6, color="#55A868",
            label="Noisy TFT (err = 0.20)")
    a2.plot(d["disc"]["exploiter"], lw=1.6, color="#C44E52",
            label="Exploiter (ALLD)")
    a2.set(xlabel="Round", ylabel=r"$\lambda$",
           title=r"(b) Behaviour-level discrimination")
    a2.legend(fontsize=8)
    a2.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures_en", "fig1_lambda_dynamics.png"),
                dpi=200)
    plt.close(fig)


def fig2(out):
    d = _ld(out, "fig2")
    if d is None:
        return
    scen = [k for k in d if k != "switches"]
    fig, axes = plt.subplots(1, len(scen), figsize=(4.2 * len(scen), 3.6),
                             sharey=True)
    for ax, sc in zip(np.atleast_1d(axes), scen):
        v = d[sc]
        ax.plot(v["lam"], lw=1.4, color="#4C72B0", label=r"own $\lambda$")
        ax.plot(v["lam_j"], lw=1.4, color="#C44E52",
                label=r"inferred $\hat\lambda_j$")
        for s in d["switches"]:
            ax.axvline(s, color="k", lw=0.8, ls=":", alpha=0.6)
        ax.set(xlabel="Round", title=SCEN_EN.get(sc, sc))
    np.atleast_1d(axes)[0].set_ylabel("Trait value")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Trait-switch tracking (dotted lines: true switches)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures_en", "fig2_trait_switch.png"),
                dpi=200)
    plt.close(fig)


def fig3(out):
    d = _ld(out, "fig3")
    if d is None:
        return
    M, names = _reorder6(d["M"], d["agents"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = _heat(ax, M, names, names, "viridis",
               "Expected payoff (row agent vs column partner),\n"
               "stationary PD (CI = 0.4)")
    ax.set_xlabel("Partner")
    ax.set_ylabel("Focal agent")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean payoff / round")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures_en",
                             "fig3_payoff_matrix_stationary.png"), dpi=200)
    plt.close(fig)


def fig4(out):
    d = _ld(out, "fig4")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(14, 3.2))
    for r, tr in d.items():
        ax.plot(tr, lw=1.5, label=NAME_EN.get(r, r))
    ax.set(xlabel="Round", ylabel="Cooperation index CI",
           title="Non-stationary payoff schedules "
                 "(CI restricted to [0, 1]; period 200 rounds)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, ncol=5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures_en", "fig4_ci_schedules.png"),
                dpi=200)
    plt.close(fig)


def fig5(out):
    d = _ld(out, "fig5")
    if d is None:
        return
    agents = d["agents"]
    regs = list(d["regimes"])
    Ms = {r: np.asarray(d["regimes"][r], float) for r in regs}
    avg = np.nanmean(np.stack(list(Ms.values())), axis=0)
    avg_r, names = _reorder6(avg, agents)
    hr = agents.index("halloreg")
    fx = [a for a in agents if a != "halloreg"]
    fxn = [NAME_EN[a] for a in fx]
    # D[r, s] = mean over the five fixed opponents j of
    #           payoff(HalloReg vs j) - payoff(strategy s vs j) in regime r.
    D = np.array([[np.nanmean([Ms[r][hr, agents.index(j)]
                               - Ms[r][agents.index(s), agents.index(j)]
                               for j in fx]) for s in fx] for r in regs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    im1 = _heat(a1, avg_r, names, names, "viridis",
                "Mean expected payoff across\nnon-stationary regimes")
    a1.set_xlabel("Partner")
    a1.set_ylabel("Focal agent")
    fig.colorbar(im1, ax=a1, shrink=0.85)
    im2 = _heat(a2, D, fxn, [NAME_EN.get(r, r) for r in regs], "RdBu_r",
                "Average payoff advantage of HalloReg\nover each fixed"
                " strategy, per regime", fmt="{:+.2f}", center=0.0)
    a2.set_xlabel("Fixed strategy")
    a2.set_ylabel("Payoff regime")
    fig.colorbar(im2, ax=a2, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures_en",
                             "fig5_payoff_nonstationary.png"), dpi=200)
    plt.close(fig)


def fig6(out):
    d = _ld(out, "fig6")
    if d is None:
        return
    regs = [NAME_EN.get(r, r) for r in d["regimes"]]
    reps = [NAME_EN[a] for a in d["replaced"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = _heat(ax, d["delta_pct"], reps, regs, "RdBu_r",
               "Group mutual-cooperation change (%) when one fixed\n"
               "strategy is replaced by a HalloReg agent",
               fmt="{:+.1f}", center=0.0)
    ax.set_xlabel("Replaced strategy")
    ax.set_ylabel("Payoff regime")
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\Delta$ mutual coop. (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures_en",
                             "fig6_population_replacement.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_figs")
    p.add_argument("--figs", default="1,2,3,4,5,6")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "figures_en"), exist_ok=True)
    for f in [s.strip() for s in args.figs.split(",") if s.strip()]:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4,
         "5": fig5, "6": fig6}[f](args.out)
        print(f"[fig{f}] rendered")


if __name__ == "__main__":
    main()
