"""
experiments.h2b_counterfactual
==============================

**H2B — what does endogenous lambda regulation contribute that a
fixed lambda cannot?**  (v3.9.5)

Three HalloReg conditions, identical in every other respect:

    reg   : lambda regulated  (lambda = clip(lambda* + phi - 0.5, 0, 1))
    lam0  : lambda held at 0  (purely self-interested valuation)
    lam1  : lambda held at 1  (purely other-regarding valuation)

crossed with five opponents: TFT, GTFT, WSLS, a second HalloReg of the
same condition (HR), and ALLD.

Pre-registered predictions (from the v3.9.4 mechanism probe, 6 seeds,
800 rounds — directional only until this experiment confirms them):

  * **Onset.** Under lam0 cooperation does not form with
    reciprocators (Z-tilde is on-policy: without lambda the first
    cooperative moves are never made, so A^Z(s) never turns
    positive). lam1 and reg both form cooperation.
  * **Withdrawal.** Under lam1 the agent keeps cooperating with ALLD
    and is exploited; reg lowers lambda and defends like lam0.
  * Therefore only reg achieves both, and the dual-context worst case
    min{payoff vs GTFT, payoff vs ALLD} is highest for reg.

Confirmatory tests (Holm-corrected together with the other
hypotheses):

    H2B-1  CC rate vs GTFT:      reg > lam0            (onset)
    H2B-2  payoff vs ALLD:       reg > lam1            (withdrawal)
    H2B-3  worst case min(GTFT, ALLD) payoff: reg > lam0
    H2B-4  worst case min(GTFT, ALLD) payoff: reg > lam1

GTFT is the confirmatory cooperative context (as in H2). lam0 versus
reg *against WSLS* is reported as exploratory and is expected to go
the **other way** (lam0 exploits WSLS's deterministic alternation and
earns more) — this is reported, not hidden.

Withdrawal mechanism (exploratory, reg condition vs ALLD)
---------------------------------------------------------
lambda = clip(lambda* + phi - 0.5). Both terms are logged per round
(``lam_l0`` and ``allo_phi``), so the drop from the first regulated
round to the withdrawal round can be split exactly:

    d(lambda*+phi) = d(lambda*) + d(phi)

The drop is measured from the early **peak** of lambda (see
``_withdrawal``). Two single-channel reconstructions are evaluated on
the logged series: hold phi at its peak value and let lambda* move
(``hold_phi`` = lambda* alone), and vice versa (``hold_l0`` = phi
alone). If neither alone reaches the withdrawal criterion, both
channels are necessary. ``t80_*`` gives the round (after the peak) at
which each channel completed 80% of its drop — the fast/slow ordering.
Withdrawal round = first round after the peak from which lambda stays
below ``WITHDRAW_LEVEL`` for ``WITHDRAW_HOLD`` consecutive rounds.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, effect_size_paired, fmt_es, one_sample_perm, perm_test,
)
from AIF_IPD.ipd.sim import run_many
from .common import Config, Registry, save_fig, save_json

LOGGER = get_logger("HalloReg.H2B")

CONDS: List[tuple] = [("reg", None), ("lam0", 0.0), ("lam1", 1.0)]
COND_LABEL = {"reg": "regulated", "lam0": "lambda=0 fixed",
              "lam1": "lambda=1 fixed"}
OPPS: List[str] = ["tft", "gtft", "wsls", "hr", "alld"]
OPP_LABEL = {"tft": "TFT", "gtft": "GTFT", "wsls": "WSLS",
             "hr": "HalloReg", "alld": "ALLD"}
COOP_CTX = "gtft"          # confirmatory cooperative context
EXPL_CTX = "alld"          # exploiter context
WITHDRAW_LEVEL = 0.10
WITHDRAW_HOLD = 20
SUSTAIN_CC = 5
PEAK_SMOOTH = 10
PEAK_HORIZON = 300


def _first_sustained(x: np.ndarray, pred, k: int) -> float:
    """First index i such that pred holds on x[i:i+k]; nan if never."""
    m = pred(x).astype(bool)
    if len(m) < k:
        return float("nan")
    run = np.convolve(m, np.ones(k, dtype=int), mode="valid")
    idx = np.flatnonzero(run == k)
    return float(idx[0]) if idx.size else float("nan")


def _withdrawal(lam: np.ndarray, l0: np.ndarray, phi: np.ndarray) -> Dict:
    """Decompose the lambda drop from its early peak to the withdrawal
    round.

    Measured shape vs ALLD (v3.9.5 probe): lambda starts at 0 (lambda*
    = 0.5 by the degeneracy guard, phi = 0), spikes to ~0.9 within ~10
    rounds because phi saturates at 1 while E_t is still optimistic,
    then falls. The decomposition is therefore anchored at the **peak**
    of the (10-round smoothed) lambda, not at the first regulated
    round — from the first round every channel starts at zero and the
    split is meaningless.
    """
    ok = np.isfinite(l0) & np.isfinite(phi)
    if not ok.any():
        return {"t_w": float("nan")}
    t0 = int(np.flatnonzero(ok)[0])
    phic = np.clip(phi, 0.0, 1.0)
    k = np.ones(PEAK_SMOOTH) / PEAK_SMOOTH
    sm = np.convolve(np.nan_to_num(lam), k, mode="same")
    horizon = min(len(lam), PEAK_HORIZON)
    tp = int(t0 + np.argmax(sm[t0:horizon]))
    t_w = _first_sustained(lam[tp:], lambda v: v < WITHDRAW_LEVEL,
                           WITHDRAW_HOLD)
    if not np.isfinite(t_w):
        return {"t_w": t_w, "t0": t0, "t_peak": tp, "lam_peak": float(lam[tp])}
    t_w = int(t_w) + tp
    d_l0 = float(l0[t_w] - l0[tp])
    d_phi = float(phic[t_w] - phic[tp])
    # Single-channel reconstructions from the peak: hold one channel
    # at its peak value and let the other follow its logged path.
    hold_phi = np.clip(l0 + phic[tp] - 0.5, 0.0, 1.0)     # lambda* alone
    hold_l0 = np.clip(l0[tp] + phic - 0.5, 0.0, 1.0)      # phi alone
    # Time at which each channel has completed 80% of its drop.
    def _t80(x, t_from, t_to):
        tot = x[t_to] - x[t_from]
        if abs(tot) < 1e-9:
            return float("nan")
        idx = np.flatnonzero((x[t_from:t_to + 1] - x[t_from]) / tot >= 0.8)
        return float(idx[0]) if idx.size else float("nan")
    return {
        "t0": t0, "t_peak": tp, "t_w": t_w,
        "lam_peak": float(lam[tp]), "lam_tw": float(lam[t_w]),
        "l0_peak": float(l0[tp]), "l0_tw": float(l0[t_w]),
        "phi_peak": float(phic[tp]), "phi_tw": float(phic[t_w]),
        "d_l0": d_l0, "d_phi": d_phi,
        "frac_l0": (d_l0 / (d_l0 + d_phi)) if abs(d_l0 + d_phi) > 1e-9
                   else float("nan"),
        "t80_l0": _t80(l0, tp, t_w), "t80_phi": _t80(phic, tp, t_w),
        "hold_phi_late": float(np.mean(hold_phi[t_w:])),
        "hold_l0_late": float(np.mean(hold_l0[t_w:])),
        "t_w_hold_phi": _first_sustained(hold_phi[tp:],
                                         lambda v: v < WITHDRAW_LEVEL,
                                         WITHDRAW_HOLD),
        "t_w_hold_l0": _first_sustained(hold_l0[tp:],
                                        lambda v: v < WITHDRAW_LEVEL,
                                        WITHDRAW_HOLD),
    }


def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H2B] lambda counterfactuals: regulated vs fixed 0 / 1")
    hk = cfg.halloreg_kwargs()
    ev = int(cfg.eval_from) if getattr(cfg, "eval_from", 0) > 0 else cfg.rounds // 2

    specs, registry = [], {}
    for ci, (cname, lf) in enumerate(CONDS):
        for oi, opp in enumerate(OPPS):
            for sd in range(cfg.seeds):
                a = {"type": "halloreg", "seed": 12_000 + sd * 41 + ci, **hk}
                if lf is not None:
                    a["lam_fixed"] = float(lf)
                if opp == "hr":
                    o = {"type": "halloreg", "seed": 12_500 + sd * 41 + ci, **hk}
                    if lf is not None:
                        o["lam_fixed"] = float(lf)
                else:
                    o = {"type": "strategy", "kind": opp,
                         "seed": 13_000 + sd * 41 + oi}
                registry[(cname, opp, sd)] = len(specs)
                specs.append({"agent": a, "opponent": o,
                              "env_err_agent": cfg.env_error,
                              "env_err_opponent": cfg.env_error,
                              # CRN across conditions at equal (seed, opp)
                              "noise_seed": 720_000 + sd * 107 + oi * 11})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H2B counterfactual dyads")

    cc = {c: {} for c, _ in CONDS}; dd = {c: {} for c, _ in CONDS}
    pay = {c: {} for c, _ in CONDS}; lam_late = {c: {} for c, _ in CONDS}
    t_cc = {c: {} for c, _ in CONDS}
    lam_traces = {}
    decomp = []
    for cname, _ in CONDS:
        for opp in OPPS:
            arr_cc, arr_dd, arr_pay, arr_lam, arr_t = [], [], [], [], []
            traces = []
            for sd in range(cfg.seeds):
                r = res[registry[(cname, opp, sd)]]
                st = np.asarray(r["hist"]["state"])
                lam = np.asarray(r["agent_log"]["lam"], dtype=float)
                arr_cc.append(float(np.mean(st[ev:] == 0)))
                arr_dd.append(float(np.mean(st[ev:] == 3)))
                arr_pay.append(float(np.mean(r["hist"]["my_payoff"][ev:])))
                arr_lam.append(float(np.mean(lam[ev:])))
                arr_t.append(_first_sustained(st, lambda v: v == 0, SUSTAIN_CC))
                traces.append(lam)
                if cname == "reg" and opp == EXPL_CTX:
                    d = _withdrawal(
                        lam, np.asarray(r["agent_log"]["lam_l0"], dtype=float),
                        np.asarray(r["agent_log"]["allo_phi"], dtype=float))
                    d["seed"] = sd
                    decomp.append(d)
            cc[cname][opp] = np.array(arr_cc); dd[cname][opp] = np.array(arr_dd)
            pay[cname][opp] = np.array(arr_pay); lam_late[cname][opp] = np.array(arr_lam)
            t_cc[cname][opp] = np.array(arr_t)
            lam_traces[(cname, opp)] = np.vstack(traces)

    # ---- Confirmatory ----
    t = perm_test(cc["reg"][COOP_CTX], cc["lam0"][COOP_CTX], paired=True,
                  alternative="greater")
    e = effect_size_paired(cc["reg"][COOP_CTX] - cc["lam0"][COOP_CTX])
    reg.confirm("H2B", "onset: CC rate vs GTFT, regulated > lambda=0",
                t["p"], direction_ok=bool(t["observed"] > 0),
                effect=f"Δ={t['observed']:+.3f}, {fmt_es(e, 'dz')}")

    t = perm_test(pay["reg"][EXPL_CTX], pay["lam1"][EXPL_CTX], paired=True,
                  alternative="greater")
    e = effect_size_paired(pay["reg"][EXPL_CTX] - pay["lam1"][EXPL_CTX])
    reg.confirm("H2B", "withdrawal: payoff vs ALLD, regulated > lambda=1",
                t["p"], direction_ok=bool(t["observed"] > 0),
                effect=f"Δ={t['observed']:+.3f}, {fmt_es(e, 'dz')}")

    worst = {c: np.minimum(pay[c][COOP_CTX], pay[c][EXPL_CTX]) for c, _ in CONDS}
    for other in ("lam0", "lam1"):
        t = perm_test(worst["reg"], worst[other], paired=True,
                      alternative="greater")
        e = effect_size_paired(worst["reg"] - worst[other])
        reg.confirm("H2B", f"dual-context worst case: regulated > "
                    f"{COND_LABEL[other]}", t["p"],
                    direction_ok=bool(t["observed"] > 0),
                    effect=f"Δ={t['observed']:+.3f}, {fmt_es(e, 'dz')}")

    # ---- Exploratory: every opponent, both metrics, both contrasts ----
    for opp in OPPS:
        for other in ("lam0", "lam1"):
            for name, tab in (("CC rate", cc), ("payoff", pay)):
                tt = perm_test(tab["reg"][opp], tab[other][opp], paired=True)
                reg.explore("H2B", f"[{OPP_LABEL[opp]}] {name}: regulated "
                                   f"vs {COND_LABEL[other]}", tt["p"],
                            effect=f"Δ={tt['observed']:+.3f}")
    # Time to first sustained CC (finite seeds only; count of failures
    # reported in the effect string).
    for opp in ("tft", "gtft", "wsls", "hr"):
        a_, b_ = t_cc["reg"][opp], t_cc["lam0"][opp]
        fin = np.isfinite(a_) & np.isfinite(b_)
        if fin.sum() >= 3:
            tt = perm_test(a_[fin], b_[fin], paired=True, alternative="less")
            p_ = tt["p"]; eff = f"Δt={tt['observed']:+.1f}R"
        else:
            p_ = 1.0; eff = "too few paired finite seeds"
        reg.explore("H2B", f"[{OPP_LABEL[opp]}] rounds to sustained CC: "
                           f"regulated < lambda=0 "
                           f"(lambda=0 never: {int(np.sum(~np.isfinite(b_)))}"
                           f"/{cfg.seeds})", p_, effect=eff)

    # ---- Exploratory: withdrawal decomposition vs ALLD ----
    fin = [d for d in decomp if np.isfinite(d.get("t_w", np.nan)) and "d_l0" in d]
    mech = {"n_withdrawn": len(fin), "n_seeds": cfg.seeds,
            "withdraw_level": WITHDRAW_LEVEL, "withdraw_hold": WITHDRAW_HOLD}
    if fin:
        for k in ("t_peak", "t_w", "lam_peak", "lam_tw", "l0_peak", "l0_tw",
                  "phi_peak", "phi_tw", "d_l0", "d_phi", "frac_l0",
                  "t80_l0", "t80_phi", "hold_phi_late", "hold_l0_late"):
            v = np.array([d[k] for d in fin], dtype=float)
            mech[k] = boot_mean_ci(v[np.isfinite(v)]) if np.isfinite(v).any() \
                else {"mean": float("nan"), "ci": [float("nan")] * 2}
        for k in ("t_w_hold_phi", "t_w_hold_l0"):
            v = np.array([d[k] for d in fin], dtype=float)
            mech[k + "_n_finite"] = int(np.isfinite(v).sum())
            mech[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        # Which channel explains more of the drop? one-sample test on
        # frac_l0 against 0.5.
        fr = np.array([d["frac_l0"] for d in fin], dtype=float)
        fr = fr[np.isfinite(fr)]
        if fr.size >= 3:
            tt = one_sample_perm(fr, 0.5)
            ci = boot_mean_ci(fr)
            reg.explore("H2B", "withdrawal vs ALLD: share of lambda drop "
                               "carried by lambda* (vs phi), test against 1/2",
                        tt["p"],
                        effect=f"frac(λ*)={ci['mean']:.2f} "
                               f"[{ci['ci'][0]:.2f}, {ci['ci'][1]:.2f}], "
                               f"n={fr.size}")
        # Does each single channel alone reproduce withdrawal?
        for k, lab in (("t_w_hold_phi", "lambda* alone (phi held)"),
                       ("t_w_hold_l0", "phi alone (lambda* held)")):
            v = np.array([d[k] for d in fin], dtype=float)
            late = np.array([d["hold_phi_late" if k.endswith("phi")
                               else "hold_l0_late"] for d in fin])
            reg.explore("H2B", f"withdrawal vs ALLD reproduced by {lab}: "
                               f"{int(np.isfinite(v).sum())}/{len(fin)} seeds",
                        1.0 if not np.isfinite(v).any() else 0.0,
                        effect=f"mean late lambda={np.mean(late):.3f}")
    else:
        reg.explore("H2B", "withdrawal vs ALLD: no seed reached the "
                           "withdrawal criterion", 1.0, effect="—")

    out = {"conditions": [c for c, _ in CONDS], "opponents": OPPS,
           "eval_from": ev, "cc": cc, "dd": dd, "payoff": pay,
           "lam_late": lam_late, "t_sustained_cc": t_cc, "worst": worst,
           "withdrawal": {"per_seed": decomp, "summary": mech}}
    _plot(cfg, out, lam_traces)
    save_json(out, cfg.results / "H2B.json")
    return out


# ============================================================== Figures
def _plot(cfg: Config, out: dict, lam_traces: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import band_plot

    colors = {"reg": "#DA8BC3", "lam0": "#4C72B0", "lam1": "#DD8452"}
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

    for k, (metric, ttl) in enumerate((("cc", "CC rate (eval window)"),
                                        ("payoff", "payoff per round (eval window)"))):
        ax = fig.add_subplot(gs[0, k])
        x = np.arange(len(OPPS)); w = 0.26
        for j, (c, _) in enumerate(CONDS):
            m = [np.mean(out[metric][c][o]) for o in OPPS]
            s = [np.std(out[metric][c][o], ddof=1) / np.sqrt(len(out[metric][c][o]))
                 if len(out[metric][c][o]) > 1 else 0.0 for o in OPPS]
            ax.bar(x + (j - 1) * w, m, w, yerr=s, color=colors[c],
                   label=COND_LABEL[c], capsize=2)
        ax.set_xticks(x); ax.set_xticklabels([OPP_LABEL[o] for o in OPPS])
        ax.set_title(f"({'ab'[k]}) {ttl}", fontsize=10)
        if k == 0:
            ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(CONDS))
    ax.bar(x, [np.mean(out["worst"][c]) for c, _ in CONDS],
           color=[colors[c] for c, _ in CONDS])
    ax.set_xticks(x); ax.set_xticklabels([COND_LABEL[c] for c, _ in CONDS],
                                         fontsize=8)
    ax.set_title("(c) dual-context worst case min(GTFT, ALLD)", fontsize=10)

    for k, opp in enumerate(("gtft", "alld")):
        ax = fig.add_subplot(gs[1, k])
        for c, _ in CONDS:
            band_plot(ax, lam_traces[(c, opp)], color=colors[c],
                      label=COND_LABEL[c])
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("round"); ax.set_ylabel("lambda")
        ax.set_title(f"({'de'[k]}) lambda trajectory vs {OPP_LABEL[opp]}",
                     fontsize=10)
        if k == 0:
            ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    fin = [d for d in out["withdrawal"]["per_seed"] if "d_l0" in d]
    if fin:
        ax.bar([0, 1], [np.mean([d["d_l0"] for d in fin]),
                        np.mean([d["d_phi"] for d in fin])],
               color=["#937860", "#55A868"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Δ lambda*", "Δ phi"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"(f) lambda drop vs ALLD split by channel "
                     f"(n={len(fin)})", fontsize=10)
    else:
        ax.text(0.5, 0.5, "no withdrawal", ha="center", va="center")
        ax.set_title("(f) lambda drop vs ALLD split by channel", fontsize=10)
    save_fig(fig, cfg, "H2B_counterfactual")
