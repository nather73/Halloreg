"""
experiments.h2_protection
=========================

**H2 — can HalloReg protect its own payoff against exploiters?**
**H2A — can HalloReg tell an exploiter from a noisy TFT?**

H2 design — self-protection
---------------------------
Main opponent: ALLD (deliberate exploiter, zero execution noise).
Two families of controls:

  (1) **Fixed-lambda empathic agents** (lambda in {0.0, 0.4, 0.8}) —
      the control that exhibits the original paper's limitation
      directly: higher lambda weights the partner's welfare, so the
      agent keeps cooperating with an exploiter and loses payoff.
      lambda = 0.4 is the reference setting of Albarracin et al. and
      serves as the **confirmatory** control; 0.0 and 0.8 are
      exploratory.

  (2) **Fixed strategies** (TFT, GTFT, WSLS, ALLC) — the standard
      benchmark against ALLD, with ALLC as the **floor condition**.

[The key contrast — dual-context worst case]
Because a higher lambda is more exploitable, fixing lambda low is
advantageous in context E — but only if one already knew the partner
was an exploiter and tuned lambda accordingly. So every agent is
exposed to **both contexts**,

    context E : ALLD (exploiter)          — self-protection needed
    context C : GTFT (generous cooperator) — mutual cooperation needed

and the comparison is on the **worst-context payoff**
min{mu_E, mu_C}.

[Choice of confirmatory controls — fixed in advance, with reasons]
The confirmatory controls are the empathic agents of Albarracin et
al. (lambda = 0.4, 0.8) and the fixed strategies. lambda = 0.0 is
**exploratory**, not confirmatory, because:

  - lambda = 0.0 is a purely self-interested active-inference agent
    with no empathy weighting — a **different model**, not the
    empathic model the original paper proposed, hence an
    inappropriate control for this hypothesis.
  - More importantly, lambda = 0.0 is **optimal by definition** in
    the exploiter context (the only force sustaining cooperation is
    removed). Taking a control that is optimal by construction as the
    confirmatory bar would fail every adaptive model.

**So that this decision cannot hide an unfavourable result**, the
lambda = 0.0 control is always **reported explicitly** as an
exploratory result. In this implementation the lambda = 0.0 agent in
fact achieves high cooperation in context C as well, because of
**reciprocity propagation inside the rollout** (the planner uses
rho-hat to compute that cooperating now is returned next round). In
other words the instrumental value of cooperation is carried by the
EFE, not by lambda, and lambda retains its construct meaning as the
weight placed on the partner's welfare as such. This is by design,
not a defect — but it forbids any overclaim of the form "cooperation
is impossible without empathy".

  - Confirmatory H2-1 : context-E payoff — HalloReg > fixed
                        lambda = 0.4 (the original paper's setting).
  - Confirmatory H2-2 : context-E payoff — HalloReg > ALLC (floor).
  - Confirmatory H2-3 : dual-context worst case — HalloReg > fixed
                        lambda = 0.4.
  - Confirmatory H2-4 : dual-context worst case — HalloReg > fixed
                        lambda = 0.8.
  - Exploratory       : the **lambda = 0.0 control**, context-wise
                        divergence of the lambda trajectory, the
                        fixed-strategy benchmarks.

H2A design — exploiter vs noisy TFT
-----------------------------------
Both opponents defect often; they differ in the *cause*:

  - ALLD      : the cause is a trait (very low alpha); defection is
                deterministic, so beta-hat is high.
  - noisy TFT : the cause is execution noise (error = 0.20);
                defection is stochastic, so beta-hat is low.

For this distinction to hold, the inference must **separate the alpha
axis (trait) from the beta axis (precision)**, so the index has two
levels:

  (i)  **Representational** : discrimination in theta-hat space,
       classifying (alpha-hat, beta-hat) with a half-split
       cross-validation. Accuracy > 0.5 means the distinction exists
       at the level of representation.
  (ii) **Behavioural** : the difference in final lambda and in
       cooperation rate. A distinction confined to representation
       without behavioural consequence has no functional meaning:
       lambda should recover against a noisy TFT and stay suppressed
       against ALLD.

  - Confirmatory H2A-1 : theta-hat discrimination accuracy > 0.5.
  - Confirmatory H2A-2 : lambda_final(noisy TFT) >
                         lambda_final(ALLD) — behavioural level.
  - Exploratory        : beta-hat, alpha-hat and late cooperation
                         differences.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, effect_size, effect_size_paired, fmt_es, one_sample_perm,
    perm_test,
)
from AIF_IPD.ipd.sim import run_many
from .common import Config, Registry, TYPE_COLORS, save_fig, save_json
from .h1_intent import DiagGaussianClassifier

LOGGER = get_logger("HalloReg.H2")

#: Lambda grid of the fixed-lambda controls
FIXED_LAMS = (0.0, 0.4, 0.8)
#: The reference lambda used as the confirmatory control
#: (the setting of Albarracin et al.)
REFERENCE_LAM = 0.4
#: Fixed-strategy benchmarks
BENCH = ("tft", "gtft", "wsls", "allc")
#: Dual contexts — E (exploitation) / C (cooperation)
CONTEXTS = {"E": {"kind": "alld", "error": 0.0},
            "C": {"kind": "gtft", "error": 0.0}}
CONTEXT_LABEL = {"E": "context E — exploiter (ALLD)",
                 "C": "context C — cooperator (GTFT)"}
#: Display labels
COND_LABEL = {"halloreg": "HalloReg", "tft": "TFT", "gtft": "GTFT",
              "wsls": "WSLS", "allc": "ALLC"}


def _cond_label(c: str) -> str:
    """Condition key -> display name."""
    if c.startswith("fixed_lam"):
        return "fixed lambda=" + c.replace("fixed_lam", "")
    return COND_LABEL.get(c, c)


# ==================================================================== H2
def run_protection(cfg: Config, reg: Registry) -> dict:
    """H2 — self-protection against exploiters and the dual-context
    worst case."""
    hk = cfg.halloreg_kwargs()
    ek = {"n_particles": cfg.n_particles, "planning_horizon": cfg.horizon}

    conditions: List[tuple] = [("halloreg", {"type": "halloreg", **hk})]
    for lam in FIXED_LAMS:
        conditions.append((f"fixed_lam{lam:.1f}",
                           {"type": "empathic", "lam": float(lam), **ek}))
    for b in BENCH:
        conditions.append((b, {"type": "strategy", "kind": b}))

    specs, registry = [], {}
    for gi, (gkey, ospec) in enumerate(CONTEXTS.items()):
        for ci, (cname, cspec) in enumerate(conditions):
            for sd in range(cfg.seeds):
                a = dict(cspec); a["seed"] = 9000 + sd * 31 + ci
                o = dict(ospec); o["type"] = "strategy"
                o["seed"] = 9500 + sd * 31 + gi
                registry[(gkey, ci, sd)] = len(specs)
                specs.append({
                    "agent": a, "opponent": o,
                    "env_err_agent": cfg.env_error,
                    "env_err_opponent": cfg.env_error,
                    # Common random numbers: identical noise
                    # realisations across conditions at the same seed, so
                    # differences come only from policy, not luck.
                    "noise_seed": 700_000 + sd * 101 + gi * 7})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H2 dual-context dyads")

    names = [c for c, _ in conditions]
    payoff = {g: {} for g in CONTEXTS}
    coop = {g: {} for g in CONTEXTS}
    lam_traces = {}
    for gi, gkey in enumerate(CONTEXTS):
        for ci, cname in enumerate(names):
            payoff[gkey][cname] = np.array([
                float(np.mean(res[registry[(gkey, ci, sd)]]["hist"]["my_payoff"]))
                for sd in range(cfg.seeds)])
            coop[gkey][cname] = np.array([
                float(np.mean(res[registry[(gkey, ci, sd)]]["hist"]["my_act"] == 0))
                for sd in range(cfg.seeds)])
            if cname == "halloreg" or cname.startswith("fixed_lam"):
                lam_traces[(gkey, cname)] = np.vstack([
                    np.asarray(res[registry[(gkey, ci, sd)]]["agent_log"]["lam"])
                    for sd in range(cfg.seeds)])

    ref = f"fixed_lam{REFERENCE_LAM:.1f}"

    # ---- Confirmatory H2-1: context E vs the reference control ----
    t1 = perm_test(payoff["E"]["halloreg"], payoff["E"][ref], paired=True,
                   alternative="greater")
    e1 = effect_size_paired(payoff["E"]["halloreg"] - payoff["E"][ref])
    reg.confirm("H2", f"payoff vs exploiter: HalloReg > fixed "
                f"lambda={REFERENCE_LAM}",
                t1["p"], direction_ok=bool(t1["observed"] > 0),
                effect=f"Δ={t1['observed']:+.3f}, {fmt_es(e1, 'dz')}")

    # ---- Confirmatory H2-2: floor condition ----
    t2 = perm_test(payoff["E"]["halloreg"], payoff["E"]["allc"], paired=True,
                   alternative="greater")
    e2 = effect_size_paired(payoff["E"]["halloreg"] - payoff["E"]["allc"])
    reg.confirm("H2", "payoff vs exploiter: HalloReg > ALLC (floor)",
                t2["p"],
                direction_ok=bool(t2["observed"] > 0),
                effect=f"Δ={t2['observed']:+.3f}, {fmt_es(e2, 'dz')}")

    # ---- Confirmatory H2-3/H2-4: dual-context worst case, against
    # the empathic controls (lambda > 0). lambda = 0.0 is a purely
    # self-interested model, not an empathic one, so it is separated
    # out as exploratory (see the module docstring).
    worst = {c: np.minimum(payoff["E"][c], payoff["C"][c]) for c in names}
    for lam in FIXED_LAMS:
        key = f"fixed_lam{lam:.1f}"
        t3 = perm_test(worst["halloreg"], worst[key], paired=True,
                       alternative="greater")
        e3 = effect_size_paired(worst["halloreg"] - worst[key])
        if lam > 0.0:
            reg.confirm("H2", f"dual-context worst case: HalloReg > "
                        f"fixed lambda={lam}", t3["p"],
                        direction_ok=bool(t3["observed"] > 0),
                        effect=f"Δ={t3['observed']:+.3f}, {fmt_es(e3, 'dz')}")
        else:
            reg.explore("H2", f"dual-context worst case: HalloReg vs "
                              f"fixed lambda={lam} (purely self-interested "
                              f"model — optimal by definition vs an "
                              f"exploiter)",
                        t3["p"],
                        effect=f"Δ={t3['observed']:+.3f}, {fmt_es(e3, 'dz')}")
    # The lambda = 0 control in context E is reported explicitly too
    # (nothing is hidden).
    t0 = perm_test(payoff["E"]["halloreg"], payoff["E"]["fixed_lam0.0"],
                   paired=True)
    reg.explore("H2", "payoff vs exploiter: HalloReg vs fixed "
                      "lambda=0.0 (empathy-free ceiling)", t0["p"],
                effect=f"Δ={t0['observed']:+.3f}")

    # ---- Exploratory: fixed-strategy benchmarks in both contexts ----
    for gkey in CONTEXTS:
        for b in BENCH:
            tb = perm_test(payoff[gkey]["halloreg"], payoff[gkey][b],
                           paired=True)
            reg.explore("H2", f"[{CONTEXT_LABEL[gkey]}] payoff: HalloReg "
                              f"vs {COND_LABEL[b]}", tb["p"],
                        effect=f"Δ={tb['observed']:+.3f}")

    # ---- Exploratory: lambda fall/rise (first vs last quarter) ----
    q = max(cfg.rounds // 4, 1)
    for gkey, direction in (("E", "greater"), ("C", "less")):
        tr = lam_traces[(gkey, "halloreg")]
        drop = tr[:, :q].mean(axis=1) - tr[:, -q:].mean(axis=1)
        td = one_sample_perm(drop, 0.0, alternative=direction)
        cd = boot_mean_ci(drop)
        arrow = "fall" if gkey == "E" else "rise"
        reg.explore("H2", f"[{CONTEXT_LABEL[gkey]}] lambda {arrow} "
                          f"(early - late)",
                    td["p"],
                    effect=f"Δλ={cd['mean']:+.3f} "
                           f"[{cd['ci'][0]:+.3f}, {cd['ci'][1]:+.3f}]")

    return {"conditions": names, "payoff": payoff, "coop": coop,
            "worst": worst, "lam_traces": lam_traces, "reference": ref}


# ==================================================================== H2A
def run_discrimination(cfg: Config, reg: Registry) -> dict:
    """H2A — telling a deliberate exploiter from a noisy
    cooperator."""
    hk = cfg.halloreg_kwargs()
    conds = [("exploiter", {"type": "strategy", "kind": "alld", "error": 0.0}),
             ("noisy_tft", {"type": "strategy", "kind": "tft", "error": 0.20})]

    specs, registry = [], {}
    for ci, (cname, ospec) in enumerate(conds):
        for sd in range(cfg.seeds):
            o = dict(ospec); o["seed"] = 11_000 + sd * 37 + ci
            registry[(ci, sd)] = len(specs)
            specs.append({
                "agent": {"type": "halloreg", "seed": 10_000 + sd * 37, **hk},
                "opponent": o,
                # Environment noise is 0: the noise to be discriminated
                # is the opponent's **internal execution noise**, and
                # adding environment noise would give both conditions
                # noise and blur the contrast.
                "env_err_agent": 0.0, "env_err_opponent": 0.0,
                "noise_seed": 710_000 + sd * 103})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H2A discrimination dyads")

    feats = ("alpha", "beta", "rho", "lambda_j")
    X = {c: np.zeros((cfg.seeds, len(feats))) for c, _ in conds}
    lam_final = {c: np.zeros(cfg.seeds) for c, _ in conds}
    coop_late = {c: np.zeros(cfg.seeds) for c, _ in conds}
    lam_traces = {c: np.zeros((cfg.seeds, cfg.rounds)) for c, _ in conds}
    # If eval_from is set it defines the 'post-learning' window
    # (default: the second half).
    half = (int(cfg.eval_from) if getattr(cfg, "eval_from", 0) > 0
            else cfg.rounds // 2)

    for ci, (cname, _) in enumerate(conds):
        for sd in range(cfg.seeds):
            r = res[registry[(ci, sd)]]
            log = r["agent_log"]
            for d, ax in enumerate(feats):
                X[cname][sd, d] = float(log[f"E_{ax}"][-1])
            lam_final[cname][sd] = float(log["lam"][-1])
            lam_traces[cname][sd] = np.asarray(log["lam"], dtype=float)
            coop_late[cname][sd] = float(
                np.mean(r["hist"]["my_act"][half:] == 0))

    # ---- Confirmatory H2A-1: theta-hat accuracy > 0.5
    # (half-split cross-validation) ----
    n_cal = cfg.seeds // 2
    names = [c for c, _ in conds]
    Xc = np.vstack([X[c][:n_cal] for c in names])
    yc = [c for c in names for _ in range(n_cal)]
    Xt = np.vstack([X[c][n_cal:] for c in names])
    n_te = cfg.seeds - n_cal
    yt = np.array([i for i in range(len(names)) for _ in range(n_te)])

    mu, sd_ = Xc.mean(axis=0), np.maximum(Xc.std(axis=0, ddof=1), 1e-6)
    clf = DiagGaussianClassifier().fit((Xc - mu) / sd_, yc, names)
    pred = clf.predict((Xt - mu) / sd_)
    acc = float(np.mean(pred == yt))

    rng = np.random.default_rng(21)
    null = np.array([np.mean(rng.permutation(pred) == yt) for _ in range(5000)])
    p_acc = (np.sum(null >= acc) + 1) / (5000 + 1)
    reg.confirm("H2A", "theta-hat discrimination accuracy > chance "
                "(0.5)", p_acc,
                direction_ok=bool(acc > 0.5),
                effect=f"acc={acc:.3f} (n={len(yt)})",
                detail={"accuracy": acc})

    # ---- Confirmatory H2A-2: behavioural discrimination (final
    # lambda) ----
    t2 = perm_test(lam_final["noisy_tft"], lam_final["exploiter"],
                   paired=True, alternative="greater")
    e2 = effect_size_paired(lam_final["noisy_tft"] - lam_final["exploiter"])
    reg.confirm("H2A", "lambda_final: noisy TFT > exploiter", t2["p"],
                direction_ok=bool(t2["observed"] > 0),
                effect=f"Δλ={t2['observed']:+.3f}, {fmt_es(e2, 'dz')}")

    # ---- Exploratory: per-axis differences and late cooperation ----
    for d, ax in enumerate(feats):
        tt = perm_test(X["noisy_tft"][:, d], X["exploiter"][:, d], paired=True)
        ee = effect_size(X["noisy_tft"][:, d], X["exploiter"][:, d])
        reg.explore("H2A", f"theta-hat axis {ax}: noisy TFT vs "
                    f"exploiter", tt["p"],
                    effect=f"Δ={tt['observed']:+.3f}, {fmt_es(ee)}")
    tc = perm_test(coop_late["noisy_tft"], coop_late["exploiter"], paired=True,
                   alternative="greater")
    reg.explore("H2A", "late cooperation rate: noisy TFT > exploiter",
                tc["p"],
                effect=f"Δ={tc['observed']:+.3f}")

    return {"features": list(feats), "theta": X, "lam_final": lam_final,
            "lam_traces": lam_traces, "coop_late": coop_late,
            "accuracy": acc, "conditions": names}


# =================================================================== Run
def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H2/H2A] defence against exploiters and "
                "noise-intent discrimination")
    a = run_protection(cfg, reg)
    b = run_discrimination(cfg, reg)
    out = {"protection": a, "discrimination": b}
    _plot(cfg, a, b)
    save_json(out, cfg.results / "H2.json")
    return out


# ============================================================== Figures
def _plot(cfg: Config, a: dict, b: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import annotate_n, band_plot, bar_with_ci

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30)

    conds = a["conditions"]
    labels = [_cond_label(c) for c in conds]
    colors = ["#DA8BC3" if c == "halloreg"
              else ("#8C8C8C" if c.startswith("fixed") else "#4C72B0")
              for c in conds]

    # (a) mean payoff per context
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(conds)); w = 0.38
    for off, gkey, col, lab in (
            (-w / 2, "E", "#937860", "context E — exploiter"),
            (+w / 2, "C", "#55A868", "context C — cooperator")):
        vals = [a["payoff"][gkey][c].mean() for c in conds]
        errs = [boot_mean_ci(a["payoff"][gkey][c])["ci"] for c in conds]
        lo = [v - e[0] for v, e in zip(vals, errs)]
        hi = [e[1] - v for v, e in zip(vals, errs)]
        ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=2.5, color=col,
               label=lab, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right",
                                         fontsize=7)
    ax.set_ylabel("mean payoff per round")
    ax.set_title("(a) H2 — payoff by context")
    ax.legend(fontsize=7); annotate_n(ax, cfg.seeds)

    # (b) dual-context worst case — the key contrast
    ax = fig.add_subplot(gs[0, 1])
    means = [a["worst"][c].mean() for c in conds]
    cis = [boot_mean_ci(a["worst"][c])["ci"] for c in conds]
    bar_with_ci(ax, labels, means, cis, colors=colors,
                ylabel="min{context E, context C} payoff", rotate=40)
    ax.set_title("(b) dual-context worst case\n"
                 "no fixed lambda satisfies both contexts at once")
    annotate_n(ax, cfg.seeds)

    # (c) lambda trajectories by context (HalloReg vs fixed lambda)
    ax = fig.add_subplot(gs[0, 2])
    band_plot(ax, a["lam_traces"][("E", "halloreg")], color="#937860",
              label="HalloReg — context E")
    band_plot(ax, a["lam_traces"][("C", "halloreg")], color="#55A868",
              label="HalloReg — context C")
    for lam in FIXED_LAMS:
        ax.axhline(lam, color="#BBBBBB", ls=":", lw=1)
    ax.text(1, FIXED_LAMS[-1] + 0.02, "fixed-lambda control levels",
            fontsize=6.5,
            color="#888888")
    ax.set_xlabel("round"); ax.set_ylabel("empathy weight lambda")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(c) context-dependent divergence of lambda")
    ax.legend(fontsize=7); annotate_n(ax, cfg.seeds)

    # (d) H2A — theta-hat scatter (alpha-hat x beta-hat)
    ax = fig.add_subplot(gs[1, 0])
    fi_a = b["features"].index("alpha")
    fi_b = b["features"].index("beta")
    for c, col, lab in (
            ("exploiter", "#937860", "deliberate exploiter (ALLD)"),
            ("noisy_tft", "#55A868", "noisy TFT (err=0.20)")):
        ax.scatter(b["theta"][c][:, fi_a], b["theta"][c][:, fi_b],
                   s=14, alpha=0.55, color=col, label=lab)
    ax.set_xlabel("alpha-hat (cooperation bias)")
    ax.set_ylabel("beta-hat (action precision)")
    ax.set_title(f"(d) H2A — representational discrimination\n"
                 f"accuracy={b['accuracy']:.3f}")
    ax.legend(fontsize=7); annotate_n(ax, cfg.seeds)

    # (e) lambda trajectory comparison
    ax = fig.add_subplot(gs[1, 1])
    band_plot(ax, b["lam_traces"]["exploiter"], color="#937860",
              label="vs exploiter")
    band_plot(ax, b["lam_traces"]["noisy_tft"], color="#55A868",
              label="vs noisy TFT")
    ax.set_xlabel("round"); ax.set_ylabel("empathy weight lambda")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(e) behavioural discrimination — lambda divergence")
    ax.legend(fontsize=7); annotate_n(ax, cfg.seeds)

    # (f) final lambda and late cooperation rate
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(2)
    w = 0.36
    lm = [b["lam_final"]["exploiter"].mean(), b["lam_final"]["noisy_tft"].mean()]
    lc = [boot_mean_ci(b["lam_final"][c])["ci"]
          for c in ("exploiter", "noisy_tft")]
    cm = [b["coop_late"]["exploiter"].mean(), b["coop_late"]["noisy_tft"].mean()]
    cc = [boot_mean_ci(b["coop_late"][c])["ci"]
          for c in ("exploiter", "noisy_tft")]
    ax.bar(x - w / 2, lm, w, yerr=[[m - c[0] for m, c in zip(lm, lc)],
                                   [c[1] - m for m, c in zip(lm, lc)]],
           capsize=3, color="#DA8BC3", label="final lambda", alpha=0.9)
    ax.bar(x + w / 2, cm, w, yerr=[[m - c[0] for m, c in zip(cm, cc)],
                                   [c[1] - m for m, c in zip(cm, cc)]],
           capsize=3, color="#4C72B0", label="late cooperation",
           alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["exploiter", "noisy TFT"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("value")
    ax.set_title("(f) behavioural consequence of the discrimination")
    ax.legend(fontsize=7)

    fig.suptitle("H2 / H2A — self-protection against exploiters and "
                 "noise-intent discrimination",
                 fontsize=12, y=0.98)
    save_fig(fig, cfg, "H2_protection_discrimination")
